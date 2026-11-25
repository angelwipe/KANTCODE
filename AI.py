import json
import sys
import time
import re
from pathlib import Path
from typing import List, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Настройки подключения и путей
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "local-model"  # LM Studio подставит активную модель

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "vacancies.json"       # <-- Укажите точное имя вашего файла
OUTPUT_FILE = BASE_DIR / "audit_results.json"

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# 2. Строгая схема ответа от ИИ
class CriterionEvaluation(BaseModel):
    criterion: Literal[
        "Название должности",
        "Обязанности",
        "Требования",
        "Оплата",
        "График",
        "Место работы",
        "Тип занятости"
    ]
    status: Literal["ОК", "Требует уточнения", "Противоречие"]
    analysis: str = Field(
        description="Анализ с учетом всех полей API и текста. Без придирок к отсутствию дублирования."
    )
    recommendation: Optional[str] = Field(
        description="Конкретный вопрос для уточнения у работодателя. Если статус 'ОК', указать null."
    )

class VacancyAuditResult(BaseModel):
    vacancy_id: str
    job_name: str
    evaluations: List[CriterionEvaluation]
    general_summary: str = Field(
        description="Краткий итог для соискателя (1-2 предложения) о прозрачности вакансии."
    )

# 3. Системный промпт
SYSTEM_PROMPT = """
Ты — аналитик качества вакансий для государственных и муниципальных организаций.
Оцени вакансию строго по 7 пунктам: Название должности, Обязанности, Требования, Оплата, График, Место работы, Тип занятости.

КАК ЧИТАТЬ ДАННЫЕ ЭТОГО ИСТОЧНИКА:
- Оплата: `salary_min`, `salary_max`, `currency` (если указаны числа — это конкретика, в тексте дублировать не нужно).
- График и занятость: `schedule` и `employment`.
- Обязанности: `duty`.
- Требования: совокупность структурированного блока `requirement` (education, experience, qualification) и текстового поля `requirements`. Обращай внимание, если в `qualification` попала информация не по теме (например, про адрес или график), или если `experience: 0` противоречит тексту `requirements`.
- Место работы: массив `addresses.address[].location` и `region.name`.

СТРОГИЕ ПРАВИЛА:
1. Учитывай и структурированные поля, и текстовые описания в совокупности. Если поле API содержит конкретные сведения, отсутствие их повторения в тексте НЕ является проблемой.
2. Отмечай только реальные пропуски, размытые формулировки (например, «5/2» без часов работы или город без улицы/формата работы) и противоречия между полями.
3. В поле `recommendation` пиши только то, что именно нужно уточнить соискателю.
4. ЗАПРЕЩЕНО: выдумывать факты, которых нет в JSON; делать выводы о нарушении ТК РФ или иных законов; оценивать форму собственности работодателя.
"""

def extract_vacancies_list(raw_data) -> list:
    """Рекурсивно находит все вакансии на любой глубине вложенности JSON."""
    found = []

    def search(node):
        if isinstance(node, dict):
            if "vacancy" in node and isinstance(node["vacancy"], dict):
                found.append(node)
            elif "job-name" in node and "id" in node:
                found.append(node)
            else:
                for value in node.values():
                    search(value)
        elif isinstance(node, list):
            for item in node:
                search(item)

    search(raw_data)
    if not found:
        raise ValueError("Не удалось найти вакансии в JSON-файле. Проверьте структуру файла.")
    return found

def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Рисует текстовый прогресс-бар для терминала."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent * 100:5.1f}% ({current}/{total})"

def audit_single_vacancy(vacancy_item: dict) -> dict:
    """Отправляет одну вакансию в LM Studio и надежно парсит JSON-ответ."""
    vac = vacancy_item.get("vacancy", vacancy_item)
    vac_id = str(vac.get("id", "UNKNOWN"))
    job_name = vac.get("job-name", "Без названия")

    # Передаем модели строгую схему прямо в промпте
    schema_instruction = (
        "\n\nВЕРНИ ОТВЕТ СТРОГО В ФОРМАТЕ JSON (без markdown-разметки и лишнего текста) ПО ЭТОЙ СХЕМЕ:\n"
        '{\n'
        '  "evaluations": [\n'
        '    {\n'
        '      "criterion": "Название должности | Обязанности | Требования | Оплата | График | Место работы | Тип занятости",\n'
        '      "status": "ОК | Требует уточнения | Противоречие",\n'
        '      "analysis": "Текст анализа",\n'
        '      "recommendation": "Что уточнить или null"\n'
        '    }\n'
        '  ],\n'
        '  "general_summary": "Краткий итог для соискателя (1-2 предложения)"\n'
        '}'
    )

    prompt = (
        f"Проведи аудит вакансии ID: {vac_id}.\n"
        f"Исходные данные JSON:\n{json.dumps(vac, ensure_ascii=False, indent=2)}"
        f"{schema_instruction}"
    )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=0.1,
    )

    raw_text = response.choices.message.content.strip()

    # Убираем блок рассуждений <think>...</think>, если модель его генерирует
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Вырезаем чистый JSON от первой '{' до последней '}'
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"Модель не вернула JSON. Ответ: {raw_text[:150]}...")

    clean_json_str = raw_text[start_idx:end_idx + 1]
    data = json.loads(clean_json_str)

    # Валидируем через Pydantic, чтобы гарантировать наличие всех нужных полей
    data["vacancy_id"] = vac_id
    data["job_name"] = job_name
    validated = VacancyAuditResult(**data)
    return validated.model_dump()

    """Отправляет одну вакансию в LM Studio и возвращает словарь с оценкой."""
    vac = vacancy_item.get("vacancy", vacancy_item)
    vac_id = str(vac.get("id", "UNKNOWN"))
    job_name = vac.get("job-name", "Без названия")

    prompt = (
        f"Проведи аудит вакансии ID: {vac_id}.\n"
        f"Исходные данные JSON:\n{json.dumps(vac, ensure_ascii=False, indent=2)}"
    )

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        response_format=VacancyAuditResult,
        temperature=0.1,
    )

    parsed_result = response.choices[0].message.parsed
    parsed_result.vacancy_id = vac_id
    parsed_result.job_name = job_name
    return parsed_result.model_dump()

def main():
    print("=" * 65)
    print("🚀 ЗАПУСК АУДИТА ВАКАНСИЙ ЧЕРЕЗ LM STUDIO")
    print("=" * 65)

    if not INPUT_FILE.exists():
        print(f"❌ Ошибка: Файл {INPUT_FILE} не найден.")
        return

    print(f"📂 Чтение файла: {INPUT_FILE.name}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    vacancies = extract_vacancies_list(raw_data)
    total = len(vacancies)
    print(f"✅ Найдено вакансий для анализа: {total}\n")

    results = []
    start_total_time = time.time()

    for idx, item in enumerate(vacancies, start=1):
        vac = item.get("vacancy", item)
        vac_id = vac.get("id", f"#{idx}")
        job_name = vac.get("job-name", "Без названия")

        print("-" * 65)
        print(f"📊 Прогресс: {render_progress_bar(idx - 1, total)}")
        print(f"⏳ [{idx}/{total}] Анализирую: {vac_id} — «{job_name}»...")
        sys.stdout.flush()

        step_start = time.time()
        try:
            audit_data = audit_single_vacancy(item)
            elapsed = time.time() - step_start
            results.append(audit_data)

            # Подсчет сводки по статусам для наглядности в консоли
            statuses = [ev["status"] for ev in audit_data["evaluations"]]
            ok_cnt = statuses.count("ОК")
            warn_cnt = statuses.count("Требует уточнения")
            err_cnt = statuses.count("Противоречие")

            print(f"✅ Готово за {elapsed:.1f} сек | Итог по блокам: 🟢 ОК: {ok_cnt} | 🟡 Уточнить: {warn_cnt} | 🔴 Противоречия: {err_cnt}")

            # Выводим краткий список проблемных мест прямо в терминал
            for ev in audit_data["evaluations"]:
                if ev["status"] != "ОК":
                    icon = "🟡" if ev["status"] == "Требует уточнения" else "🔴"
                    print(f"   {icon} {ev['criterion']}: {ev['analysis']}")

        except Exception as e:
            elapsed = time.time() - step_start
            print(f"❌ Ошибка за {elapsed:.1f} сек на вакансии {vac_id}: {e}")
            results.append({"vacancy_id": vac_id, "job_name": job_name, "error": str(e)})

        # Промежуточное сохранение после каждой вакансии (защита от потери данных при сбое)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_total_time
    print("=" * 65)
    print(f"🏁 ОБРАБОТКА ЗАВЕРШЕНА! {render_progress_bar(total, total)}")
    print(f"⏱  Общее время: {total_time:.1f} сек (в среднем {total_time / total:.1f} сек на вакансию)")
    print(f"💾 Полный отчет сохранен в: {OUTPUT_FILE}")
    print("=" * 65)

if __name__ == "__main__":
    main()
