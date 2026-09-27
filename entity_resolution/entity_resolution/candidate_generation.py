import csv
import logging
import multiprocessing as mp
import time
from pathlib import Path

from tqdm import tqdm

from .io import count_rows, iter_rows, read_tsv
from .retrieval import Retriever

log = logging.getLogger(__name__)


def _candidate_worker(rank, gpu_id, gpu_ids, config_dict, source1_path, temp_dir, total_rows):
    """Retrieve one deterministic Source1 shard using one embedding worker/GPU."""
    from .config import PipelineConfig
    config = PipelineConfig.from_mapping(config_dict)
    logging.basicConfig(level=config.log_level, format=f"%(asctime)s %(levelname)s [gpu:{gpu_id}] %(message)s")
    retriever = Retriever(config.index_dir, config, device=(f"cuda:{gpu_id}" if gpu_id is not None else None))
    logging.info("Loaded index_dir=%s countries=%d shards=%d", config.index_dir, len(retriever.countries), retriever.shard_count)
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    shard_path = Path(temp_dir) / f"candidates-gpu{gpu_id}.tsv"
    expected = (total_rows + len(gpu_ids) - 1 - rank) // len(gpu_ids)
    progress = tqdm(total=expected, desc=f"GPU {gpu_id} candidate rows", unit="rows", position=rank, disable=not config.show_progress)
    ordinal = 0
    done = 0
    try:
        with shard_path.open("w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            for rows in read_tsv(source1_path, chunk_size=config.chunk_size):
                selected = []
                selected_ordinals = []
                for item in rows:
                    if ordinal % len(gpu_ids) == rank:
                        selected.append(item)
                        selected_ordinals.append(ordinal)
                    ordinal += 1
                if selected:
                    batch_ids = retriever.retrieve_batch(selected)
                    for item_ordinal, item, ids in zip(selected_ordinals, selected, batch_ids):
                        writer.writerow([item_ordinal, item["entity_id"], ",".join(ids)])
                    done += len(selected)
                    progress.update(len(selected))
    finally:
        progress.close()
    return str(shard_path), done


def generate_candidates(config):
    """Generate candidates with one retrieval process per configured GPU.

    Workers write temporary ordinal-keyed shards; the parent merges them in
    Source1 order, so the final candidates.tsv remains deterministic.
    """
    gpu_ids = list(config.gpu_ids) or [None]
    total_rows = count_rows(config.source1_path)
    Path(config.temp_dir).mkdir(parents=True, exist_ok=True)
    config_dict = dict(config.__dict__)
    ctx = mp.get_context("spawn")
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_candidate_worker,
            args=(rank, gpu_id, gpu_ids, config_dict, config.source1_path, config.temp_dir, total_rows),
            name=f"candidate-gpu-{gpu_id}",
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    failed = [(p.name, p.exitcode) for p in processes if p.exitcode != 0]
    if failed:
        raise RuntimeError(f"Candidate worker failure(s): {failed}")

    shard_paths = [Path(config.temp_dir) / f"candidates-gpu{gpu_id}.tsv" for gpu_id in gpu_ids]
    handles = [p.open(encoding="utf-8", newline="") for p in shard_paths]
    readers = [csv.reader(h, delimiter="\t") for h in handles]
    current = []
    written = 0
    nonempty = 0
    try:
        for rank, reader in enumerate(readers):
            row = next(reader, None)
            if row is not None:
                current.append((int(row[0]), rank, row))
        current.sort()
        with open(config.candidates_path, "w", encoding="utf-8", newline="") as out:
            writer = csv.writer(out, delimiter="\t", lineterminator="\n")
            writer.writerow(["entity_id", "candidate_ids"])
            progress = tqdm(total=total_rows, desc="Merging candidates", unit="rows", disable=not config.show_progress)
            try:
                while current:
                    ordinal, rank, row = current.pop(0)
                    writer.writerow([row[1], row[2]])
                    progress.update(1)
                    written += 1
                    nonempty += bool(row[2])
                    nxt = next(readers[rank], None)
                    if nxt is not None:
                        current.append((int(nxt[0]), rank, nxt))
                        current.sort()
            finally:
                progress.close()
    finally:
        for handle in handles:
            handle.close()
        for path in shard_paths:
            path.unlink(missing_ok=True)
    if written != total_rows:
        raise RuntimeError(f"Candidate output row count mismatch: wrote {written}, expected {total_rows}")
    if total_rows and nonempty == 0:
        raise RuntimeError(
            "All generated candidate lists are empty. Check index_dir, country values, "
            "and that reference indexes were built successfully."
        )
    log.info("Candidate generation complete: rows=%d nonempty=%d (%.2f%%)", written, nonempty, 100 * nonempty / max(written, 1))


def _split(v):
    return {x for x in (v or "").split(",") if x}


def evaluate(candidates_path, truth_path, output_path):
    import json
    import os
    import sqlite3
    import statistics
    import random
    db_path = str(output_path) + ".truth.sqlite"
    db = sqlite3.connect(db_path)
    try:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("CREATE TABLE truth (entity_id TEXT PRIMARY KEY, matched TEXT NOT NULL)")
        batch = []
        for r in __import__("entity_resolution.io", fromlist=["iter_rows"]).iter_rows(
            truth_path, columns=("source1_entity_id", "matched_entity_ids"), chunk_size=10000
        ):
            batch.append((r["source1_entity_id"], r.get("matched_entity_ids") or ""))
            if len(batch) >= 10000:
                db.executemany("INSERT OR REPLACE INTO truth VALUES (?, ?)", batch)
                db.commit(); batch.clear()
        if batch:
            db.executemany("INSERT OR REPLACE INTO truth VALUES (?, ?)", batch); db.commit()
        db.execute("CREATE INDEX truth_id ON truth(entity_id)")

        # Streaming aggregates plus a bounded reservoir for approximate percentiles.
        rng = random.Random(42); sample = []; sample_limit = 100000
        rows = total_candidates = zero = empty_truth = full = 0
        recall_sum = nonempty_recall_sum = 0.0; nonempty_count = 0; max_candidates = 0
        by_matches = {}
        with open(candidates_path, encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                truth_row = db.execute("SELECT matched FROM truth WHERE entity_id=?", (r["entity_id"],)).fetchone()
                g = _split(truth_row[0]) if truth_row else set()
                got = _split(r.get("candidate_ids")); candidate_count = len(got); tp = len(g & got)
                rows += 1; total_candidates += candidate_count; max_candidates = max(max_candidates, candidate_count)
                zero += candidate_count == 0; empty_truth += not g; full += g <= got
                recall = tp / len(g) if g else 1.0; recall_sum += recall
                if g: nonempty_recall_sum += recall; nonempty_count += 1
                key = len(g); total, matched = by_matches.get(key, (0, 0.0)); by_matches[key] = (total + 1, matched + recall)
                if len(sample) < sample_limit: sample.append(candidate_count)
                else:
                    j = rng.randrange(rows)
                    if j < sample_limit: sample[j] = candidate_count
        out={"rows":rows,"mean_candidates":total_candidates/max(rows,1),
             "median_candidates":_percentile(sample,.50),"p90_candidates":_percentile(sample,.90),
             "p95_candidates":_percentile(sample,.95),"p99_candidates":_percentile(sample,.99),"max_candidates":max_candidates,
             "zero_retrieved_pct":100*zero/max(rows,1),"empty_ground_truth_pct":100*empty_truth/max(rows,1),
             "mean_recall":recall_sum/max(rows,1),
             "mean_recall_nonempty_ground_truth":nonempty_recall_sum/max(nonempty_count,1),
             "nonempty_ground_truth_rows":nonempty_count,
             "full_match_coverage_pct":100*full/max(rows,1),
             "recall_by_true_match_count":{str(k): total_recall/count for k,(count,total_recall) in by_matches.items()}}
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path,"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
        return out
    finally:
        db.close()
        try: os.unlink(db_path)
        except FileNotFoundError: pass


def _percentile(values,p):
    if not values:return 0
    a=sorted(values); return a[min(len(a)-1,int(round((len(a)-1)*p)))]
