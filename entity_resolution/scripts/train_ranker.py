#!/usr/bin/env python3
import argparse
import json
from entity_resolution.ranker import train_model

p = argparse.ArgumentParser(description="Train a CatBoost candidate-ranking model")
p.add_argument("--candidates", required=True)
p.add_argument("--ground-truth", required=True)
p.add_argument("--source1", required=True)
p.add_argument("--reference", nargs="+", required=True, help="Source2 and Source3 TSV paths")
p.add_argument("--model-out", required=True)
p.add_argument("--max-negatives", type=int, default=20)
p.add_argument("--iterations", type=int, default=500)
p.add_argument("--depth", type=int, default=8)
p.add_argument("--learning-rate", type=float, default=0.08)
p.add_argument("--task-type", choices=("CPU", "GPU"), default="CPU")
p.add_argument("--devices", default="0:1", help="CatBoost GPU devices, e.g. 0:1")
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
meta = train_model(a.candidates, a.ground_truth, a.source1, a.reference, a.model_out,
                   a.max_negatives, a.iterations, a.depth, a.learning_rate,
                   a.task_type, a.devices, a.seed)
print(json.dumps(meta, indent=2))
