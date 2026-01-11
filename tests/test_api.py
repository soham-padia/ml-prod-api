from __future__ import annotations

from fastapi.testclient import TestClient

from ml_prod_api.api import create_app
from ml_prod_api.config import Settings


def test_health_and_ready_with_stub():
    settings = Settings(app_env="test", model_provider="stub", model_version="test", artifact_dir="artifacts/dev")
    app = create_app(settings)

    with TestClient(app) as c:
        r = c.get("/healthz")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

        r = c.get("/readyz")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"


def test_predict_validation():
    settings = Settings(app_env="test", model_provider="stub", model_version="test", artifact_dir="artifacts/dev")
    app = create_app(settings)

    with TestClient(app) as c:
        r = c.post("/v1/predict", json={"text": "   "})
        assert r.status_code == 422

        r = c.post("/v1/predict", json={"text": "hello"})
        assert r.status_code == 200
        body = r.json()
        assert "prediction" in body
        assert body["prediction"]["provider"] == "stub"


def test_predict_batch_limits():
    settings = Settings(
        app_env="test",
        model_provider="stub",
        model_version="test",
        artifact_dir="artifacts/dev",
        max_batch_size=3,
    )
    app = create_app(settings)

    with TestClient(app) as c:
        r = c.post("/v1/predict_batch", json={"texts": ["a", "b", "c"]})
        assert r.status_code == 200

        r = c.post("/v1/predict_batch", json={"texts": ["a", "b", "c", "d"]})
        assert r.status_code == 422
