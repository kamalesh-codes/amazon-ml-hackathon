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


def _split(v):
    return {x for x in (v or "").split(",") if x}


def evaluate(candidates_path, truth_path, output_path):
    import json
    import statistics
    truth={r["source1_entity_id"]:_split(r["matched_entity_ids"]) for r in __import__("entity_resolution.io",fromlist=["iter_rows"]).iter_rows(truth_path,columns=("source1_entity_id","matched_entity_ids"))}
    rows=[]
    with open(candidates_path,encoding="utf-8") as f:
        for r in csv.DictReader(f,delimiter="\t"):
            g=truth.get(r["entity_id"],set()); got=_split(r.get("candidate_ids")); tp=len(g&got)
            rows.append((r["entity_id"],g,got,tp))
    counts=[len(x[2]) for x in rows]; recalls=[(x[3]/len(x[1]) if x[1] else 1.0) for x in rows]
    by_matches={}
    for _,g,_,tp in rows: by_matches.setdefault(len(g),[]).append(tp/len(g) if g else 1.0)
    out={"rows":len(rows),"mean_candidates":statistics.mean(counts) if counts else 0,"median_candidates":statistics.median(counts) if counts else 0,
         "p90_candidates":_percentile(counts,.90),"p95_candidates":_percentile(counts,.95),"p99_candidates":_percentile(counts,.99),"max_candidates":max(counts,default=0),
         "zero_retrieved_pct":100*sum(c==0 for c in counts)/max(len(rows),1),"empty_ground_truth_pct":100*sum(not x[1] for x in rows)/max(len(rows),1),
         "mean_recall":statistics.mean(recalls) if recalls else 0,"full_match_coverage_pct":100*sum(x[1]<=x[2] for x in rows)/max(len(rows),1),
         "recall_by_true_match_count":{str(k):statistics.mean(v) for k,v in by_matches.items()}}
    with open(output_path,"w",encoding="utf-8") as f: json.dump(out,f,indent=2)
    return out


def _percentile(values,p):
    if not values:return 0
    a=sorted(values); return a[min(len(a)-1,int(round((len(a)-1)*p)))]
