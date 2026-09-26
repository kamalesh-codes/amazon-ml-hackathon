import csv
from pathlib import Path
from typing import Iterator

ENTITY_COLUMNS = ("entity_id", "business_name", "business_address", "country")
GROUND_TRUTH_COLUMNS = ("source1_entity_id", "matched_entity_ids")

def read_tsv(path: str, columns=ENTITY_COLUMNS, chunk_size=4096) -> Iterator[list[dict[str, str]]]:
    """Stream a TSV in bounded chunks; missing scalar values become empty strings."""
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in columns):
            raise ValueError(f"{path}: expected columns {columns}, found {reader.fieldnames}")
        chunk = []
        for row in reader:
            chunk.append({c: (row.get(c) or "") for c in columns})
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

def iter_rows(path: str, columns=ENTITY_COLUMNS, chunk_size=4096):
    for chunk in read_tsv(path, columns, chunk_size):
        yield from chunk

def count_rows(path: str, columns=ENTITY_COLUMNS) -> int:
    return sum(1 for _ in iter_rows(path, columns, 8192))

def write_candidates(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["entity_id", "candidate_ids"])
        for entity_id, ids in rows:
            w.writerow([entity_id, ",".join(ids)])
