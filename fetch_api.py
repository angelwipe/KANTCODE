"""получение сырых вакансий. Выбор источника: API сервера или локальный JSON.
Результат в data/resultsva.json том же формате, что ответ API (results.vacancies[].vacancy).
Только стандартная библиотека. Запуск: python source.py  (меню)  или  python source.py api --max 200
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

URL = "http://opendata.trudvsem.ru/api/v1/vacancies/region/39"
OUT = "data/resultsva.json"
TEST_DIR = "test"


class FetchError(RuntimeError):
    pass


def extract(payload: dict) -> list[dict]:
    """Достаёт список вакансий из ответа API."""
    try:
        items = payload["results"]["vacancies"]
    except (KeyError, TypeError) as e:
        raise FetchError("Неизвестный формат: нет results.vacancies") from e
    return [i["vacancy"] if isinstance(i, dict) and "vacancy" in i else i for i in items]


def _get(url: str, timeout: int, retries: int) -> dict:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last = e
            time.sleep(2 ** attempt)
    raise FetchError(f"API недоступен ({url}): {last}")


def fetch_api(url=URL, max_vacancies=200, page_size=100, paging="page", timeout=30, retries=3) -> list[dict]:
    """Постранично качает вакансии. paging='page': offset=номер страницы; 'items': offset=смещение в записях."""
    out, page = [], 0
    while len(out) < max_vacancies:
        offset = page if paging == "page" else page * page_size
        payload = _get(f"{url}?offset={offset}&limit={page_size}", timeout, retries)
        batch = extract(payload)
        out += batch
        page += 1
        total = (payload.get("meta") or {}).get("total") or 0
        print(f"  страница {page}: +{len(batch)} (всего {len(out)} из {total or '?'})")
        if not batch or len(batch) < page_size or (total and page * page_size >= total):
            break
    return out[:max_vacancies]


def load_local(path: str) -> list[dict]:
    if not path.lower().endswith(".json"):
        raise FetchError(f"{path}: это не .json файл")
    if not os.path.isfile(path):
        raise FetchError(f"{path}: файл не найден")
    try:
        with open(path, encoding="utf-8") as f:
            return extract(json.load(f))
    except (OSError, json.JSONDecodeError) as e:
        raise FetchError(f"не удалось прочитать {path}: {e}") from e


def save_raw(vacancies: list[dict], path=OUT) -> None:
    """Атомарно сохраняет в формате ответа API."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"status": "200", "meta": {"total": len(vacancies)},
                   "results": {"vacancies": [{"vacancy": v} for v in vacancies]}}, f, ensure_ascii=False)
    os.replace(tmp, path)


def pick_file() -> str:
    """Список .json из папки test/ или ввод своего пути."""
    files = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith(".json")) if os.path.isdir(TEST_DIR) else []
    for i, f in enumerate(files, 1):
        print(f" {i}. {TEST_DIR}/{f}")
    print(" 0. Ввести свой путь")
    c = input("> ").strip()
    if c.isdigit() and 1 <= int(c) <= len(files):
        return os.path.join(TEST_DIR, files[int(c) - 1])
    if c == "0":
        return input("Путь к .json: ").strip().strip('"')
    raise FetchError("неверный выбор")


def menu() -> list[dict]:
    print("Источник данных:\n 1. Скачать с сервера (API)\n 2. Файл .json (папка test/ или свой путь)")
    c = input("> ").strip()
    if c == "1":
        n = input("Сколько вакансий [200]: ").strip()
        return fetch_api(max_vacancies=int(n) if n.isdigit() else 200)
    if c == "2":
        return load_local(pick_file())
    raise FetchError("Неверный выбор")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="без аргументов — меню выбора источника")
    sub = ap.add_subparsers(dest="cmd")
    a = sub.add_parser("api"); a.add_argument("--url", default=URL); a.add_argument("--max", type=int, default=200)
    a.add_argument("--page-size", type=int, default=100); a.add_argument("--paging", choices=["page", "items"], default="page")
    f = sub.add_parser("file"); f.add_argument("path")
    for p in (ap, a, f):
        p.add_argument("--out", default=OUT)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "api":
            v = fetch_api(args.url, args.max, args.page_size, args.paging)
        elif args.cmd == "file":
            v = load_local(args.path)
        else:
            v = menu()
    except FetchError as e:
        print(f"ОШИБКА: {e}. Прежний {args.out} не тронут.", file=sys.stderr)
        return 2
    save_raw(v, args.out)
    print(f"Сохранено {len(v)} вакансий → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
