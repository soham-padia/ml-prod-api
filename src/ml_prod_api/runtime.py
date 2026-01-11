from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np

from .metrics import INFERENCE_LATENCY
from .models import ModelInterface, ModelPrediction


class StubModel(ModelInterface):
    provider = "stub"

    def predict_one(self, text: str) -> ModelPrediction:
        # A deterministic tiny heuristic so local dev works without artifacts.
        lowered = text.lower()
        toxic_keywords = ("hate", "awful", "stupid", "idiot", "kill", "trash")
        score = 0.8 if any(k in lowered for k in toxic_keywords) else 0.1
        return ModelPrediction(probability_toxic=float(score))


class BaselineSklearnModel(ModelInterface):
    provider = "baseline"

    def __init__(self, artifact_dir: str):
        p = Path(artifact_dir) / "baseline" / "model.joblib"
        self.pipeline = joblib.load(p)

    def predict_one(self, text: str) -> ModelPrediction:
        start = time.perf_counter()
        prob = float(self.pipeline.predict_proba([text])[0, 1])
        dur = max(0.0, time.perf_counter() - start)
        INFERENCE_LATENCY.labels(provider=self.provider).observe(dur)
        return ModelPrediction(probability_toxic=prob)

    def predict_batch(self, texts: list[str]) -> list[ModelPrediction]:
        start = time.perf_counter()
        probs = self.pipeline.predict_proba(texts)[:, 1].astype(float).tolist()
        dur = max(0.0, time.perf_counter() - start)
        INFERENCE_LATENCY.labels(provider=self.provider).observe(dur)
        return [ModelPrediction(probability_toxic=float(p)) for p in probs]


class TransformerHFModel(ModelInterface):
    provider = "transformer"

    def __init__(self, artifact_dir: str, device: str | None = None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer  # lazy import

        model_dir = Path(artifact_dir) / "transformer"
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()

        if device is None:
            device = "cuda" if self._has_cuda() else "cpu"
        self.device = device
        self.model.to(self.device)


    @staticmethod
    def _has_cuda() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _infer_probs(self, texts: list[str]) -> list[float]:
        import torch

        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            logits = out.logits.detach().cpu().numpy()
        # Assume label 1 is toxic (binary classifier)
        # Convert logits -> softmax probability for class 1
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
        return probs[:, 1].astype(float).tolist()

    def predict_one(self, text: str) -> ModelPrediction:
        start = time.perf_counter()
        prob = self._infer_probs([text])[0]
        dur = max(0.0, time.perf_counter() - start)
        INFERENCE_LATENCY.labels(provider=self.provider).observe(dur)
        return ModelPrediction(probability_toxic=float(prob))

    def predict_batch(self, texts: list[str]) -> list[ModelPrediction]:
        start = time.perf_counter()
        probs = self._infer_probs(texts)
        dur = max(0.0, time.perf_counter() - start)
        INFERENCE_LATENCY.labels(provider=self.provider).observe(dur)
        return [ModelPrediction(probability_toxic=float(p)) for p in probs]


def load_manifest(artifact_dir: str) -> dict:
    p = Path(artifact_dir) / "manifest.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


def load_model(provider: str, artifact_dir: str) -> ModelInterface:
    if provider == "stub":
        return StubModel()
    if provider == "baseline":
        return BaselineSklearnModel(artifact_dir=artifact_dir)
    if provider == "transformer":
        return TransformerHFModel(artifact_dir=artifact_dir)
    raise ValueError(f"Unknown MODEL_PROVIDER: {provider}")
