import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Literal
from openai import OpenAI
from pydantic import BaseModel, Field

# 1. Настройки подключения к LM Studio и путей к файлам
LM_STUDIO_URL = "http://localhost:1234/v1"
MODEL_NAME = "local-model"  # LM Studio автоматически использует загруженную модель

BASE_DIR = Path(__file__).parent
INPUT_FILE = BASE_DIR / "vacancies.json"       # Имя вашего исходного файла
OUTPUT_FILE = BASE_DIR / "audit_results.json"  # Имя файла с результатами

client = OpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

# 2. Строгая схема валидации ответа ИИ
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
        description="Анализ с учетом всех полей JSON. Без придирок к отсутствию дублирования."
    )
    recommendation: Optional[str] = Field(
        description="Что именно уточнить соискателю. Если статус 'ОК', указать null."
    )

class VacancyAuditResult(BaseModel):
    vacancy_id: str
    job_name: str
    evaluations: List[CriterionEvaluation]
    general_summary: str = Field(
        description="Краткий итог для соискателя (1-2 предложения) о прозрачности вакансии."
    )
SYSTEM_PROMPT = """
Ты — эксперт по оценке качества вакансий для государственных и муниципальных организаций.
Твоя задача — оценить вакансию по 7 критериям: Название должности, Обязанности, Требования, Оплата, График, Место работы, Тип занятости.

СТРУКТУРА ВХОДНОГО JSON:
- `job_name`: название должности. Если оно слишком общее (например, просто «Специалист» без указания отдела или направления), отметь необходимость уточнить профиль работы.
- `duty`: обязанности. Отмечай размытые формулировки вроде «другие задачи по необходимости» или «выполнение поручений».
- `requirements`, `education`, `experience_years`, `qualification`: блок требований. Оценивай их в совокупности. Если `education: null` или `experience_years: 0`, а в тексте `requirements` указаны иные условия — отметь это. Обращай внимание, если в `qualification` записаны условия не о квалификации, а о графике, адресе или собеседовании.
- `salary` (`min`, `max`, `currency`, `text`): оплата труда. Если указана только нижняя граница (`max: null`, например «от 35000 RUB»), укажи, что стоит уточнить размер премий и верхний предел дохода. Если указана конкретная вилка (`min` и `max`), это конкретика — отсутствие её повторения в тексте НЕ является проблемой.
- `schedule`: график работы. Если указано только «Сменный график» или «Гибкий график» без дней и часов, отметь необходимость уточнения. Проверяй противоречия с полем `qualification`.
- `addresses` и `region_name`: место работы. Если в `addresses` указан только регион («Калининградская область») или только город без точного адреса/формата работы (офис/разъездной) — отметь это.
- `employment`: тип занятости.

СТРОГИЕ ПРАВИЛА:
1. Учитывай все поля в совокупности. Если структурированное поле содержит конкретные сведения, отсутствие их дублирования в текстовом поле НЕ считается проблемой.
2. В `recommendation` формулируй только конкретный вопрос для уточнения у работодателя. Если статус «ОК», ставь `null`.
3. ЗАПРЕЩЕНО: выдумывать факты, которых нет в JSON; делать выводы о нарушении ТК РФ или иных законов; определять и комментировать форму собственности работодателя.
"""

def extract_vacancies_list(raw_data) -> list:
    """Находит список вакансий нового формата на любой глубине вложенности JSON."""
    found = []
    def search(node):
        if isinstance(node, dict):
            if "job_name" in node and "id" in node:
                found.append(node)
            else:
                for value in node.values():
                    search(value)
        elif isinstance(node, list):
            for item in node:
                search(item)

    search(raw_data)
    if not found:
        raise ValueError("Не удалось найти вакансии с полями 'id' и 'job_name' в JSON-файле.")
    return found

def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    """Рисует прогресс-бар в консоли."""
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {percent * 100:5.1f}% ({current}/{total})"

def audit_single_vacancy(vac: dict) -> dict:
    """Отправляет вакансию в LM Studio и надежно извлекает JSON-ответ."""
    vac_id = str(vac.get("id", "UNKNOWN"))
    job_name = vac.get("job_name", "Без названия")

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
        f"Данные вакансии:\n{json.dumps(vac, ensure_ascii=False, indent=2)}"
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

    raw_text = response.choices[0].message.content.strip()

    # Очищаем от тегов <think>...</think> (если в LM Studio стоит рассуждающая модель)
    raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # Извлекаем чистый JSON-объект от первой '{' до последней '}'
    start_idx = raw_text.find("{")
    end_idx = raw_text.rfind("}")
    if start_idx == -1 or end_idx == -1:
        raise ValueError(f"Модель вернула ответ не в формате JSON: {raw_text[:150]}...")

    clean_json_str = raw_text[start_idx:end_idx + 1]
    data = json.loads(clean_json_str)

    # Привязываем точные ID и название из исходника и проверяем схему
    data["vacancy_id"] = vac_id
    data["job_name"] = job_name
    validated = VacancyAuditResult(**data)
    return validated.model_dump()
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

    for idx, vac in enumerate(vacancies, start=1):
        vac_id = str(vac.get("id", f"#{idx}"))
        job_name = vac.get("job_name", "Без названия")

        print("-" * 65)
        print(f"📊 Прогресс: {render_progress_bar(idx - 1, total)}")
        print(f"⏳ [{idx}/{total}] Анализирую: {vac_id} — «{job_name}»...")
        sys.stdout.flush()

        step_start = time.time()
        try:
            audit_data = audit_single_vacancy(vac)
            elapsed = time.time() - step_start
            results.append(audit_data)

            statuses = [ev["status"] for ev in audit_data["evaluations"]]
            ok_cnt = statuses.count("ОК")
            warn_cnt = statuses.count("Требует уточнения")
            err_cnt = statuses.count("Противоречие")

            print(f"✅ Готово за {elapsed:.1f} сек | Итог по блокам: 🟢 ОК: {ok_cnt} | 🟡 Уточнить: {warn_cnt} | 🔴 Противоречия: {err_cnt}")

            for ev in audit_data["evaluations"]:
                if ev["status"] != "ОК":
                    icon = "🟡" if ev["status"] == "Требует уточнения" else "🔴"
                    print(f"   {icon} {ev['criterion']}: {ev['analysis']}")

        except Exception as e:
            elapsed = time.time() - step_start
            print(f"❌ Ошибка за {elapsed:.1f} сек на вакансии {vac_id}: {e}")
            results.append({"vacancy_id": vac_id, "job_name": job_name, "error": str(e)})

        # Промежуточное сохранение после каждой вакансии
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
