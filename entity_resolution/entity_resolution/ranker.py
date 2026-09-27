"""Feature extraction and CatBoost candidate ranking utilities."""
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

from .io import iter_rows
from .preprocessing import normalize_address, normalize_name

FEATURE_NAMES = (
    "candidate_rank",
    "name_exact",
    "name_token_jaccard",
    "name_char_jaccard",
    "address_exact",
    "address_token_jaccard",
    "address_char_jaccard",
    "numeric_token_jaccard",
    "country_match",
    "query_address_missing",
    "candidate_address_missing",
    "name_length_delta",
    "address_length_delta",
)


def _tokens(value):
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _jaccard(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / max(len(a | b), 1)


def _char_jaccard(a, b, n=3):
    aa = {a[i:i+n] for i in range(max(0, len(a)-n+1))}
    bb = {b[i:i+n] for i in range(max(0, len(b)-n+1))}
    return _jaccard(aa, bb)


def _numeric_tokens(value):
    return set(re.findall(r"\d+", value))


def pair_features(query, candidate, rank):
    qn, cn = normalize_name(query.get("business_name")), normalize_name(candidate.get("business_name"))
    qa, ca = normalize_address(query.get("business_address")), normalize_address(candidate.get("business_address"))
    qnt, cnt = _tokens(qn), _tokens(cn)
    qat, cat = _tokens(qa), _tokens(ca)
    return [
        float(rank),
        float(bool(qn) and qn == cn),
        _jaccard(qnt, cnt),
        _char_jaccard(qn, cn),
        float(bool(qa) and qa == ca),
        _jaccard(qat, cat),
        _char_jaccard(qa, ca),
        _jaccard(_numeric_tokens(qa), _numeric_tokens(ca)),
        float(bool(query.get("country")) and bool(candidate.get("country")) and query.get("country", "").casefold() == candidate.get("country", "").casefold()),
        float(not bool(qa)),
        float(not bool(ca)),
        float(abs(len(qn) - len(cn))),
        float(abs(len(qa) - len(ca))),
    ]


def load_entities(source_paths):
    entities = {}
    for path in source_paths:
        for row in iter_rows(path):
            entities[row["entity_id"]] = row
    return entities


def load_truth(path):
    truth = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            truth[row["source1_entity_id"]] = {x for x in (row.get("matched_entity_ids") or "").split(",") if x}
    return truth


def load_candidates(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            ids = [x for x in (row.get("candidate_ids") or "").split(",") if x]
            yield row["entity_id"], ids


def make_training_data(candidate_path, query_entities, reference_entities, truth, max_negatives=20, seed=42):
    rng = random.Random(seed)
    features, labels, groups = [], [], []
    for query_id, candidate_ids in load_candidates(candidate_path):
        query = query_entities.get(query_id)
        if query is None:
            continue
        positives = truth.get(query_id, set())
        pairs = [(cid, cid in positives) for cid in candidate_ids if cid in reference_entities]
        negative_ids = [cid for cid, is_positive in pairs if not is_positive]
        keep_negatives = set(rng.sample(negative_ids, min(max_negatives, len(negative_ids))))
        selected = [(cid, is_positive) for cid, is_positive in pairs if is_positive or cid in keep_negatives]
        if not selected:
            continue
        groups.append(len(selected))
        for rank, (cid, is_positive) in enumerate(selected, 1):
            features.append(pair_features(query, reference_entities[cid], rank))
            labels.append(int(is_positive))
    if not features:
        raise ValueError("No trainable candidate pairs found; check candidate IDs and reference files")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32), groups


def train_model(candidate_path, ground_truth_path, source1_path, reference_paths, model_path,
                max_negatives=20, iterations=500, depth=8, learning_rate=0.08,
                task_type="CPU", devices="0:1", seed=42):
    from catboost import CatBoostClassifier
    queries = load_entities([source1_path])
    references = load_entities(reference_paths)
    truth = load_truth(ground_truth_path)
    x, y, groups = make_training_data(candidate_path, queries, references, truth, max_negatives, seed)
    model = CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=learning_rate,
        loss_function="Logloss", eval_metric="AUC", random_seed=seed,
        task_type=task_type, devices=devices, verbose=100,
        allow_writing_files=False,
    )
    model.fit(x, y)
    model.save_model(str(model_path))
    metadata = {
        "feature_names": list(FEATURE_NAMES), "candidate_path": str(candidate_path),
        "max_negatives": max_negatives, "rows": int(len(y)), "positive_pairs": int(y.sum()),
        "negative_pairs": int((y == 0).sum()), "groups": len(groups),
    }
    Path(str(model_path) + ".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def predict_candidates(candidate_path, source1_path, reference_paths, model_path, output_path, top_k=50, threshold=None):
    from catboost import CatBoostClassifier
    queries = load_entities([source1_path])
    references = load_entities(reference_paths)
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        writer.writerow(["entity_id", "candidate_ids"])
        for query_id, candidate_ids in load_candidates(candidate_path):
            query = queries.get(query_id)
            scored = []
            if query is not None:
                valid = [(cid, rank) for rank, cid in enumerate(candidate_ids, 1) if cid in references]
                if valid:
                    x = np.asarray([pair_features(query, references[cid], rank) for cid, rank in valid], dtype=np.float32)
                    scores = model.predict_proba(x)[:, 1]
                    scored = [(float(score), cid) for (cid, _), score in zip(valid, scores) if threshold is None or score >= threshold]
            scored.sort(reverse=True)
            ids = [cid for _, cid in scored[:top_k]]
            writer.writerow([query_id, ",".join(ids)])
            written += 1
    return written
