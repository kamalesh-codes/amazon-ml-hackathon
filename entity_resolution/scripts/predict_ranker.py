#!/usr/bin/env python3
import argparse
from entity_resolution.ranker import predict_candidates

p = argparse.ArgumentParser(description="Rerank candidate TSV rows with a saved CatBoost model")
p.add_argument("--candidates", required=True)
p.add_argument("--source1", required=True)
p.add_argument("--reference", nargs="+", required=True, help="Source2 and Source3 TSV paths")
p.add_argument("--model", required=True)
p.add_argument("--output", required=True)
p.add_argument("--top-k", type=int, default=50)
p.add_argument("--threshold", type=float, default=None)
a = p.parse_args()
rows = predict_candidates(a.candidates, a.source1, a.reference, a.model, a.output, a.top_k, a.threshold)
print(f"Wrote {rows} rows to {a.output}")
