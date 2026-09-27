#!/usr/bin/env python3
import logging
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from tqdm import tqdm

from entity_resolution.config import PipelineConfig
from entity_resolution.io import count_rows, read_tsv
from entity_resolution.preprocessing import normalize_entity
from entity_resolution.country import normalize_country
from entity_resolution.embedding import make_embedder
from entity_resolution.indexing import VectorIndex


def _new_indexes(c):
    if c.retrieval_mode == "combined":
        return {"combined": VectorIndex(c.embedding_dim, c.nlist, c.pq_m, c.pq_nbits, c.nprobe, c.exact_threshold)}
    return {
        "name": VectorIndex(c.embedding_dim, c.nlist, c.pq_m, c.pq_nbits, c.nprobe, c.exact_threshold),
        "address": VectorIndex(c.embedding_dim, c.nlist, c.pq_m, c.pq_nbits, c.nprobe, c.exact_threshold),
    }


def _build_worker(rank, gpu_id, gpu_ids, config_dict, sources, total_rows):
    """Build one independent index shard on one GPU.

    Every worker reads the reference files independently but embeds only rows whose
    global ordinal belongs to its rank. This keeps GPU memory isolated and avoids
    sharing CUDA state between processes.
    """
    c = PipelineConfig.from_mapping(config_dict)
    logging.basicConfig(level=c.log_level, format=f"%(asctime)s %(levelname)s [gpu:{gpu_id}] %(message)s")
    device = f"cuda:{gpu_id}" if gpu_id is not None else None
    logging.info("Starting worker rank=%d on device=%s", rank, device or "cpu")
    indexes = {}
    emb = make_embedder(c, device=device)
    progress = tqdm(
        total=(total_rows + len(gpu_ids) - 1 - rank) // len(gpu_ids),
        desc=f"GPU {gpu_id} indexing",
        unit="rows",
        position=rank,
        disable=not c.show_progress,
    )
    ordinal = 0
    try:
        for source in sources:
            for rows in read_tsv(source, chunk_size=c.chunk_size):
                selected = []
                for row in rows:
                    if ordinal % len(gpu_ids) == rank:
                        selected.append(row)
                    ordinal += 1
                if not selected:
                    continue
                groups = defaultdict(list)
                selected = [normalize_entity(row) for row in selected]
                for row in selected:
                    row["country"] = normalize_country(row.get("country")) or "__UNKNOWN__"
                    groups[row["country"]].append(row)
                for country, rs in groups.items():
                    indexes.setdefault(country, _new_indexes(c))
                if c.retrieval_mode == "combined":
                    vectors = emb.encode(
                        [f"{r['business_name_norm']} [SEP] {r['business_address_norm']}" for r in selected],
                        c.embedding_batch_size,
                    )
                else:
                    names = emb.encode([r["business_name_norm"] for r in selected], c.embedding_batch_size)
                    addresses = emb.encode([r["business_address_norm"] for r in selected], c.embedding_batch_size)
                # Encode the whole selected chunk per field. Calling the model once
                # per country creates tiny GPU batches and is the main throughput trap.
                for country, rs in groups.items():
                    # Rows in each country group retain their order from selected.
                    meta = [{"entity_id": r["entity_id"], "source": source, "country": country} for r in rs]
                    positions = [i for i, r in enumerate(selected) if r["country"] == country]
                    if c.retrieval_mode == "combined":
                        indexes[country]["combined"].add(vectors[positions], meta)
                    else:
                        indexes[country]["name"].add(names[positions], meta)
                        indexes[country]["address"].add(addresses[positions], meta)
                progress.update(len(selected))
    finally:
        progress.close()

    shard_dir = Path(c.index_dir) / f"shard-gpu{gpu_id}"
    for country, ix in indexes.items():
        out = shard_dir / country
        for field, vector_index in ix.items():
            vector_index.save(out, field)
    logging.info("Completed worker rank=%d: countries=%d, rows=%d", rank, len(indexes), sum(len(next(iter(ix.values())).metadata) for ix in indexes.values()))


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
    c = PipelineConfig.from_mapping(OmegaConf.to_container(cfg, resolve=True))
    logging.basicConfig(level=c.log_level, format="%(asctime)s %(levelname)s %(message)s")
    sources = (c.source2_path, c.source3_path)
    total_rows = sum(count_rows(source) for source in sources)
    gpu_ids = list(c.gpu_ids) or [None]
    logging.info("Launching %d embedding worker(s) across gpu_ids=%s", len(gpu_ids), gpu_ids)
    config_dict = dict(c.__dict__)
    ctx = mp.get_context("spawn")
    processes = []
    for rank, gpu_id in enumerate(gpu_ids):
        p = ctx.Process(
            target=_build_worker,
            args=(rank, gpu_id, gpu_ids, config_dict, sources, total_rows),
            name=f"entity-index-gpu-{gpu_id}",
        )
        p.start()
        processes.append(p)
    failed = []
    for p in processes:
        p.join()
        if p.exitcode != 0:
            failed.append((p.name, p.exitcode))
    if failed:
        raise RuntimeError(f"Index worker failure(s): {failed}")
    logging.info("All GPU index workers completed successfully")


if __name__ == "__main__":
    main()
