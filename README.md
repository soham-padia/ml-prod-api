
# ml-prod-api  
**Production-grade ML Inference API (FastAPI + Prometheus)**

A complete reference implementation of a **production-ready NLP toxicity classification service**, focused on real-world concerns such as observability, validation, security, and deployment readiness — not just model serving.

---

## Runtime & Compatibility

- **Python:** 3.11 (recommended)  
  > Python 3.13 currently has known FastAPI / Pydantic annotation edge cases.
- Tested on macOS

---

## What This Service Does

- Exposes a REST API to classify text as **toxic** or **non-toxic**
- Supports **single** and **batch** inference
- Loads models at startup and exposes readiness state
- Emits **Prometheus-compatible metrics**
- Enforces **input validation, rate limits, and optional auth**
- Designed to be containerized and deployed to cloud platforms

---

## Models

- **Baseline (lightweight):**
  - TF-IDF + Logistic Regression (scikit-learn)
- **Transformer (pluggable):**
  - HuggingFace DistilBERT (default)
  - Can be swapped for MiniLM or other encoders

Model selection is controlled via configuration (`MODEL_PROVIDER`).

---

## API Endpoints

| Method | Path | Purpose |
|------|------|--------|
| GET | `/` | Service metadata |
| POST | `/v1/predict` | Single-text inference |
| POST | `/v1/predict_batch` | Batch inference |
| GET | `/healthz` | Liveness probe |
| GET | `/readyz` | Readiness probe (model loaded) |
| GET | `/metrics` | Prometheus metrics |

---

## Example Request

```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"you are awful"}'
````

```json
{
  "prediction": {
    "label": "toxic",
    "probability_toxic": 0.8,
    "model_version": "dev",
    "provider": "stub"
  }
}
```

---

## Production Features

### Input Validation

* Max text length enforcement
* Empty input checks
* Batch size limits

### Observability

* **Structured JSON logging**

  * request_id
  * route
  * status_code
  * latency_ms
  * model_version
  * (raw input text is never logged)
* **Prometheus metrics**

  * `http_requests_total{route,status}`
  * `http_request_latency_seconds`
  * `inference_latency_seconds{provider}`

### Security

* Optional API key authentication
* Configurable CORS
* Rate limiting via SlowAPI
* Dependency pinning
* Non-root Docker container

---

## Local Development

### Install dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### Run the API

```bash
export MODEL_PROVIDER=stub
uvicorn ml_prod_api.api:app --reload --app-dir src --host 0.0.0.0 --port 8000
```

### Run tests

```bash
pytest
```

---

## Training (Optional)

Training scripts are separated from runtime dependencies.

```bash
pip install -r requirements-train.txt
python training/train_tfidf.py
```

Artifacts are saved to the configured `artifact_dir`.

---

## Docker

### Build and run

```bash
docker build -t ml-prod-api .
docker run -p 8000:8000 ml-prod-api
```

### Docker Compose (API + Prometheus)

```bash
docker-compose up
```

Prometheus UI: [http://localhost:9090](http://localhost:9090)

---

## Benchmarking

Basic load testing with `hey`:

```bash
./scripts/benchmark.sh
```

Results and methodology are documented in `BENCHMARK.md`.

---

## Deployment

The service is designed for deployment on:

* AWS ECS / Fargate
* GCP Cloud Run
* Kubernetes

See `DEPLOYMENT.md` for step-by-step instructions.

---

## Project Goals

This repository intentionally focuses on:

* **Production correctness**
* **Operational visibility**
* **Clean interfaces**
* **Failure-aware design**

It is meant as a **reference implementation**, not a toy demo.

---

## License

MIT

