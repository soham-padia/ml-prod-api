#!/usr/bin/env python3
"""
Train a baseline TF-IDF + Logistic Regression toxicity classifier.

Dataset: HuggingFace `civil_comments` (toxicity score). We convert to binary:
- toxic if toxicity >= threshold (default 0.5)
- non-toxic otherwise

Exports an intermediate model under `./.runs/<model_version>/baseline.joblib`.
Use scripts/export_artifacts.py to create the runtime artifact layout in artifacts/<model_version>/.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class TrainConfig:
    model_version: str
    threshold: float
    max_samples: int
    random_seed: int


def load_binary_dataset(threshold: float, max_samples: int, random_seed: int):
    ds = load_dataset("civil_comments", split="train")
    # Columns: text, toxicity, ... (continuous). We'll binarize.
    if max_samples > 0:
        ds = ds.shuffle(seed=random_seed).select(range(min(max_samples, len(ds))))

    texts = [t if isinstance(t, str) else "" for t in ds["text"]]
    tox = np.array(ds["toxicity"], dtype=float)
    y = (tox >= threshold).astype(int)
    return texts, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--threshold", type=float, default=float(os.getenv("TOX_THRESHOLD", "0.5")))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", "50000")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_version=args.model_version,
        threshold=args.threshold,
        max_samples=args.max_samples,
        random_seed=args.random_seed,
    )

    texts, y = load_binary_dataset(cfg.threshold, cfg.max_samples, cfg.random_seed)
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=0.2, random_state=cfg.random_seed, stratify=y
    )

    pipe: Pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000)),
            ("clf", LogisticRegression(max_iter=2000, n_jobs=None)),
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_val)[:, 1]
    preds = (probs >= 0.5).astype(int)

    auc = roc_auc_score(y_val, probs)
    report = classification_report(y_val, preds, digits=4)

    run_dir = Path(".runs") / cfg.model_version
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "baseline.joblib"
    joblib.dump(pipe, model_path)

    metrics = {
        "roc_auc": float(auc),
        "val_size": int(len(y_val)),
        "threshold": cfg.threshold,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("Saved:", model_path)
    print("Metrics:", json.dumps(metrics, indent=2))
    print("\nClassification report:\n", report)


if __name__ == "__main__":
    main()
