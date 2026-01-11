from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrediction:
    probability_toxic: float  # 0..1


class ModelInterface:
    provider: str

    def predict_one(self, text: str) -> ModelPrediction:
        raise NotImplementedError

    def predict_batch(self, texts: list[str]) -> list[ModelPrediction]:
        return [self.predict_one(t) for t in texts]
