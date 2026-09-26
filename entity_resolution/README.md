# Modular Entity Resolution Candidate Generation

Candidate generation only for multilingual Source1 → Source2+Source3 matching. It preserves every Source1 row, searches separate name/address indexes, unions both Top-K results, and never forces a single match.

## Layout

- `entity_resolution/io.py`: bounded TSV streaming and output
- `preprocessing.py`, `country.py`: Unicode-safe normalization and deterministic routing
- `embedding.py`: BGE-M3 backend; `hash` backend is only for offline tests
- `indexing.py`, `metadata.py`: FAISS IndexFlatIP for small partitions and IndexIVFPQ for large partitions, with JSONL metadata
- `retrieval.py`, `candidate_generation.py`: country routing, Top-50 name/address retrieval, union/dedup, evaluation
- `scripts/`: build indexes, generate candidates, evaluate
- `tests/`: unit and end-to-end coverage
- `config/config.yaml`: Hydra configuration used by every pipeline stage

## Small end-to-end run

Create the three source TSVs with exactly `entity_id business_name business_address country` (tab-separated). `business_address` may be blank for individual rows; blanks are normalized to an empty retrieval text and do not cause the row to be dropped. The ground-truth TSV must have exactly `source1_entity_id matched_entity_ids`, where `matched_entity_ids` is a comma-separated complete set of true reference IDs. Install dependencies and run from the project root:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. python scripts/build_indexes.py
PYTHONPATH=. python scripts/generate_candidates.py
PYTHONPATH=. python scripts/evaluate_candidates.py
pytest -q
```

All scripts use Hydra. Configuration is loaded from `config/config.yaml`; any key can be altered at run time with `key=value` overrides, without editing YAML. Examples:

```bash
# Change input/output paths
PYTHONPATH=. python scripts/build_indexes.py source2_path=/data/source2.tsv source3_path=/data/source3.tsv index_dir=/data/indexes

# Tune retrieval and memory settings
PYTHONPATH=. python scripts/generate_candidates.py top_k=100 embedding_batch_size=32 chunk_size=2048 nprobe=32

# Use BGE-M3 instead of the offline hash test backend and select GPUs
PYTHONPATH=. python scripts/build_indexes.py backend=sentence_transformers gpu_ids='[0,1]' embedding_batch_size=16

# Evaluate alternate files
PYTHONPATH=. python scripts/evaluate_candidates.py candidates_path=outputs/candidates.tsv ground_truth_path=data/ground_truth.tsv evaluation_path=outputs/evaluation.json

# Print the fully resolved Hydra configuration
PYTHONPATH=. python scripts/generate_candidates.py --cfg job
```

Hydra overrides are typed: use `top_k=100` for integers, `fallback_mode=none` for strings, and `gpu_ids='[0,1]'` for lists. The default `config.yaml` uses `backend: hash` so the small pipeline can run offline; production runs should override it with `backend=sentence_transformers` after installing PyTorch, Sentence-Transformers, and a FAISS build.

For production set `backend` to `sentence_transformers`, install the requirements, ensure BGE-M3 is available, and use `gpu_ids: [0,1]`. `chunk_size` and `embedding_batch_size` are the main memory controls. Reference vectors are embedded once, added incrementally, and persisted by country; candidate output is streamed. Large partitions use IVF-PQ (`nlist`, `pq_m`, `pq_nbits`, `nprobe`); small partitions use exact inner product. Metadata is separate from FAISS and preserves source/entity IDs.

## Resource and resilience notes

The implementation never loads a complete TSV into a DataFrame and never materializes all candidate pairs. It uses normalized embeddings and inner product (cosine equivalent). Build indexes is a separate reusable stage. Completed country directories can be reused; to resume a partial build, run into a new index directory and atomically rename after successful completion. Candidate generation writes rows incrementally; for interruption-safe production runs, execute per Source1 shard and concatenate only after validating headers and non-overlapping IDs.

This initial implementation intentionally does not include a supervised reranker. Ground-truth evaluation reports mean/median/p90/p95/p99/max candidate count, zero-retrieval rate, empty-ground-truth rate, mean recall, full-match coverage, and recall by true-match count.
