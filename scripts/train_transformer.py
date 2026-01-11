#!/usr/bin/env python3
"""
Fine-tune a HuggingFace transformer for binary toxicity classification.

Default base model: distilbert-base-uncased

Dataset: HuggingFace `civil_comments` (toxicity score -> binary with threshold)

Outputs:
- HuggingFace model + tokenizer in `./.runs/<model_version>/transformer/`
- Training metrics in `./.runs/<model_version>/metrics.json`
Use scripts/export_artifacts.py to package into `artifacts/<model_version>/`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from evaluate import load as load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


@dataclass(frozen=True)
class TrainConfig:
    model_version: str
    base_model: str
    threshold: float
    max_samples: int
    max_length: int
    random_seed: int
    epochs: int
    batch_size: int
    learning_rate: float


def make_binary(ds, threshold: float):
    def _map(ex):
        tox = float(ex.get("toxicity", 0.0))
        ex["label"] = int(tox >= threshold)
        ex["text"] = ex.get("text") if isinstance(ex.get("text"), str) else ""
        return ex

    return ds.map(_map)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--base-model", default=os.getenv("BASE_MODEL", "distilbert-base-uncased"))
    parser.add_argument("--threshold", type=float, default=float(os.getenv("TOX_THRESHOLD", "0.5")))
    parser.add_argument("--max-samples", type=int, default=int(os.getenv("MAX_SAMPLES", "20000")))
    parser.add_argument("--max-length", type=int, default=int(os.getenv("MAX_LENGTH", "256")))
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "1")))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BATCH_SIZE", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LR", "2e-5")))
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        model_version=args.model_version,
        base_model=args.base_model,
        threshold=args.threshold,
        max_samples=args.max_samples,
        max_length=args.max_length,
        random_seed=args.random_seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    torch.manual_seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)

    ds = load_dataset("civil_comments", split="train")
    if cfg.max_samples > 0:
        ds = ds.shuffle(seed=cfg.random_seed).select(range(min(cfg.max_samples, len(ds))))

    ds = make_binary(ds, cfg.threshold)
    ds = ds.train_test_split(test_size=0.2, seed=cfg.random_seed, stratify_by_column="label")
    train_ds = ds["train"]
    eval_ds = ds["test"]

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_length)

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[c for c in train_ds.column_names if c not in ("label",)])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=[c for c in eval_ds.column_names if c not in ("label",)])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2)

    accuracy = load_metric("accuracy")
    f1 = load_metric("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
            "f1": f1.compute(predictions=preds, references=labels, average="binary")["f1"],
        }

    out_dir = Path(".runs") / cfg.model_version / "transformer"
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=0.01,
        logging_steps=50,
        seed=cfg.random_seed,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    eval_result = trainer.evaluate()

    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    run_dir = Path(".runs") / cfg.model_version
    metrics = {
        "train": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in train_result.metrics.items()},
        "eval": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in eval_result.items()},
        "base_model": cfg.base_model,
        "threshold": cfg.threshold,
        "max_length": cfg.max_length,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "train_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("Saved transformer model to:", out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
