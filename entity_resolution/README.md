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

### Progress bars and GPU visibility

Reference indexing and candidate generation show `tqdm` progress bars with total rows and throughput. They can be disabled for log files or batch schedulers:

```bash
PYTHONPATH=. python scripts/build_indexes.py show_progress=false
PYTHONPATH=. python scripts/generate_candidates.py show_progress=false
```

The index builder now launches one spawned process per configured GPU. Each worker loads its own BGE-M3 model on its assigned GPU, reads the reference files, embeds only its deterministic row shard, and writes indexes under `index_dir/shard-gpu<id>/`. Retrieval automatically discovers every shard and searches all of them before unioning candidates. This avoids duplicating one giant reference embedding matrix in a single GPU process. Each GPU will show activity as its worker processes embedding batches; very small datasets may still finish before utilization graphs show sustained load.

Candidate generation also launches one process per configured GPU. Each process owns a BGE-M3 model, receives a deterministic Source1 row shard, embeds names and addresses for an entire input chunk, and writes a temporary ordinal-keyed shard. The parent merges those shards back into Source1 order. Thus both index building and candidate generation use all configured GPUs:

```bash
PYTHONPATH=. python scripts/build_indexes.py \
  backend=sentence_transformers gpu_ids='[0,1]' \
  chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=256 use_fp16=true

PYTHONPATH=. python scripts/generate_candidates.py \
  backend=sentence_transformers gpu_ids='[0,1]' \
  chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=256 use_fp16=true
```

The main speed controls are `embedding_batch_size`, `chunk_size`, `max_seq_length`, and `use_fp16`. The aggressive defaults use 2,048 texts per model batch, 32,768 input rows per worker chunk, a 256-token maximum, and fp16 on CUDA. These settings change batching and padding only; BGE-M3, 1024-dimensional vectors, separate name/address embeddings, Top-K, country blocking, and candidate union semantics are unchanged. If a worker runs out of GPU memory, reduce `embedding_batch_size` in this order: 1024, 512, 256. If very long addresses are common, use `max_seq_length=512` to avoid truncating their tail. Runtime depends on GPU model, model download/cache state, FAISS build, country distribution, and disk speed, so benchmark a representative 100K-row sample before committing to a three-hour SLA.

If the quality-profile run still misses the deadline, use the explicit fast profile. It uses the multilingual 384-dimensional `paraphrase-multilingual-MiniLM-L12-v2` encoder instead of BGE-M3 and one combined `business_name + business_address` embedding pass per row. This changes embeddings and retrieval from separate name/address indexes to a combined index, so build a separate index directory and evaluate it before using the output:

```bash
PYTHONPATH=. python scripts/build_indexes.py \
  profile=fast index_dir=indexes_fast gpu_ids='[0,1]' \
  chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=128 use_fp16=true

PYTHONPATH=. python scripts/generate_candidates.py \
  profile=fast index_dir=indexes_fast gpu_ids='[0,1]' \
  chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=128 use_fp16=true

PYTHONPATH=. python scripts/evaluate_candidates.py \
  candidates_path=outputs/candidates.tsv \
  ground_truth_path=data/ground_truth.tsv \
  evaluation_path=outputs/evaluation_fast.json
```

For the Kaggle training files, use the same `profile=fast`, `backend=sentence_transformers`, and `index_dir` in all three stages. The reference files must be supplied during index building; otherwise Hydra falls back to the local `data/` defaults:

```bash
DATA=/kaggle/input/datasets/kamaleshcodes/amazon-ml-hackathon/student_resource/dataset/train

PYTHONPATH=. python scripts/build_indexes.py \
  profile=fast backend=sentence_transformers \
  source2_path=$DATA/train_source2.tsv \
  source3_path=$DATA/train_source3.tsv \
  index_dir=indexes_fast gpu_ids='[0,1]' \
  chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=128 use_fp16=true

PYTHONPATH=. python scripts/generate_candidates.py \
  profile=fast backend=sentence_transformers \
  source1_path=$DATA/train_source1.tsv \
  index_dir=indexes_fast candidates_path=outputs/candidates.tsv \
  top_k=50 chunk_size=32768 embedding_batch_size=2048 \
  max_seq_length=128 use_fp16=true gpu_ids='[0,1]'

PYTHONPATH=. python scripts/evaluate_candidates.py \
  candidates_path=outputs/candidates.tsv \
  ground_truth_path=$DATA/train_ground_truth.tsv \
  evaluation_path=outputs/evaluation_fast.json
```

For example, with `gpu_ids='[0,1]'` the output layout is:

```text
indexes/
├── shard-gpu0/IN/name.index
├── shard-gpu0/IN/address.index
├── shard-gpu1/IN/name.index
└── shard-gpu1/IN/address.index
```

For production set `backend` to `sentence_transformers`, install the requirements, ensure BGE-M3 is available, and use `gpu_ids: [0,1]`. `chunk_size` and `embedding_batch_size` are the main memory controls. Reference vectors are embedded once, added incrementally, and persisted by country; candidate output is streamed. Large partitions use IVF-PQ (`nlist`, `pq_m`, `pq_nbits`, `nprobe`); small partitions use exact inner product. Metadata is separate from FAISS and preserves source/entity IDs.

## Resource and resilience notes

The implementation never loads a complete TSV into a DataFrame and never materializes all candidate pairs. It uses normalized embeddings and inner product (cosine equivalent). Build indexes is a separate reusable stage. Completed country directories can be reused; to resume a partial build, run into a new index directory and atomically rename after successful completion. Candidate generation writes rows incrementally; for interruption-safe production runs, execute per Source1 shard and concatenate only after validating headers and non-overlapping IDs.

This initial implementation intentionally does not include a supervised reranker. Ground-truth evaluation reports mean/median/p90/p95/p99/max candidate count, zero-retrieval rate, empty-ground-truth rate, mean recall, full-match coverage, and recall by true-match count.

Candidate generation now fails fast if no country indexes are found, if the output row count differs from Source1, or if every candidate list is empty. Worker logs show the loaded `index_dir`, country count, and shard count. Evaluation also reports `mean_recall_nonempty_ground_truth`, which avoids confusing the no-match rows with actual retrieval recall.

Evaluation is designed for Kaggle-scale files: it stores the ground-truth lookup in a temporary SQLite file and streams `candidates.tsv` one row at a time. It does not keep all candidate rows, candidate sets, or metric arrays in RAM. Median and percentile candidate counts use a bounded 100,000-row reservoir sample, while row counts, means, recall, and coverage are streaming aggregates. The temporary `.truth.sqlite` file is deleted after evaluation.

## Train and use the candidate-ranking model

Candidate generation creates a retrieval set; the ranker scores those candidates and cannot recover a true match absent from the candidate TSV. Training requires the candidate TSV, ground truth TSV, Source1 TSV, and the Source2/Source3 reference TSVs because pair features are computed from names, addresses, countries, and candidate rank. The trainer keeps every positive pair and samples up to `--max-negatives` incorrect pairs per Source1 row. This controls the size of the training set while preserving hard, high-ranked negatives.

Install the model dependency:

```bash
python -m pip install -r requirements.txt
```

Train on CPU:

```bash
PYTHONPATH=. python scripts/train_ranker.py \
  --candidates outputs/candidates.tsv \
  --ground-truth /path/to/train_ground_truth.tsv \
  --source1 /path/to/train_source1.tsv \
  --reference /path/to/train_source2.tsv /path/to/train_source3.tsv \
  --model-out outputs/candidate_ranker.cbm \
  --max-negatives 20 --iterations 500
```

Train with both Kaggle GPUs using CatBoost:

```bash
PYTHONPATH=. python scripts/train_ranker.py \
  --candidates outputs/candidates.tsv \
  --ground-truth $DATA/train_ground_truth.tsv \
  --source1 $DATA/train_source1.tsv \
  --reference $DATA/train_source2.tsv $DATA/train_source3.tsv \
  --model-out outputs/candidate_ranker.cbm \
  --max-negatives 20 --iterations 500 \
  --task-type GPU --devices 0:1
```

Use the saved model to produce a reranked candidate file:

```bash
PYTHONPATH=. python scripts/predict_ranker.py \
  --candidates outputs/candidates.tsv \
  --source1 $DATA/train_source1.tsv \
  --reference $DATA/train_source2.tsv $DATA/train_source3.tsv \
  --model outputs/candidate_ranker.cbm \
  --output outputs/candidates_reranked.tsv \
  --top-k 50
```

Evaluate the reranked output with the existing evaluator:

```bash
PYTHONPATH=. python scripts/evaluate_candidates.py \
  candidates_path=outputs/candidates_reranked.tsv \
  ground_truth_path=$DATA/train_ground_truth.tsv \
  evaluation_path=outputs/evaluation_reranked.json
```

The saved `.cbm` model is accompanied by `.cbm.json` metadata containing feature names and training-pair counts. Use `--threshold 0.5` (or calibrate another threshold on validation data) only when variable candidate-list lengths are preferred; without a threshold, prediction returns the top `--top-k` candidates for every Source1 row.
