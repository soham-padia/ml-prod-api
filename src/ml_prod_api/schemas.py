from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    text: str = Field(..., description="Input text to classify")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if v is None:
            raise ValueError("text is required")
        s = v.strip()
        if not s:
            raise ValueError("text must not be empty")
        return s


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="Batch of texts")

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v: List[str]) -> List[str]:
        if v is None or len(v) == 0:
            raise ValueError("texts must not be empty")
        cleaned = []
        for t in v:
            if t is None:
                raise ValueError("batch contains null text")
            s = str(t).strip()
            if not s:
                raise ValueError("batch contains empty/whitespace text")
            cleaned.append(s)
        return cleaned


class Prediction(BaseModel):
    label: str = Field(..., description="toxic | non_toxic")
    probability_toxic: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    provider: str


class PredictResponse(BaseModel):
    prediction: Prediction


class PredictBatchResponse(BaseModel):
    predictions: List[Prediction]


class HealthResponse(BaseModel):
    status: str


class RootResponse(BaseModel):
    name: str
    version: str
    model_version: str
    provider: str
