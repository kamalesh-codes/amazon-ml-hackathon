#!/usr/bin/env python3
import logging
from collections import defaultdict
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from entity_resolution.config import PipelineConfig
from entity_resolution.io import read_tsv
from entity_resolution.preprocessing import normalize_entity
from entity_resolution.country import normalize_country
from entity_resolution.embedding import make_embedder
from entity_resolution.indexing import VectorIndex

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg):
 c=PipelineConfig.from_mapping(OmegaConf.to_container(cfg, resolve=True))
 logging.basicConfig(level=c.log_level,format="%(asctime)s %(levelname)s %(message)s")
 # Two passes are intentionally avoided: collect only per-country index objects; vector batches are released after add.
 indexes={}; emb=make_embedder(c, device=(f"cuda:{c.gpu_ids[0]}" if c.gpu_ids else None))
 for source in (c.source2_path,c.source3_path):
  for rows in read_tsv(source,chunk_size=c.chunk_size):
   groups=defaultdict(list)
   for row in rows:
    row=normalize_entity(row); row["country"]=normalize_country(row.get("country")) or "__UNKNOWN__"; groups[row["country"]].append(row)
   for country, rs in groups.items():
    if country not in indexes: indexes[country]={"name":VectorIndex(c.embedding_dim,c.nlist,c.pq_m,c.pq_nbits,c.nprobe,c.exact_threshold),"address":VectorIndex(c.embedding_dim,c.nlist,c.pq_m,c.pq_nbits,c.nprobe,c.exact_threshold)}
    for field in ("name","address"):
     key="business_name_norm" if field=="name" else "business_address_norm"; vec=emb.encode([r[key] for r in rs],c.embedding_batch_size)
     meta=[{"entity_id":r["entity_id"],"source":source,"country":country} for r in rs]; indexes[country][field].add(vec,meta)
 for country, ix in indexes.items():
  out=Path(c.index_dir)/country; ix["name"].save(out,"name"); ix["address"].save(out,"address")
  logging.info("country=%s vectors=%d",country,len(ix["name"].metadata))
if __name__=="__main__": main()
