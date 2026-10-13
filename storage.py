"""Storage minimale: un file JSON per documento, in storage/<doc_id>/analysis.json.
Per una versione production andrebbe sostituito con un vero database (es. Postgres)."""

import json
from pathlib import Path

STORAGE_DIR = Path(__file__).parent / "storage"


def save_analysis(doc_id: str, analysis: dict, user_id: str) -> None:
    doc_dir = STORAGE_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    payload = {**analysis, "user_id": user_id}
    with open(doc_dir / "analysis.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_analysis(doc_id: str) -> dict | None:
    path = STORAGE_DIR / doc_id / "analysis.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
