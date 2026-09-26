import json
from pathlib import Path

def save_metadata(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

def load_metadata(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
