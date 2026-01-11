from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# Cardinality note:
# - route should be a templated path (FastAPI route path), not raw URL.
# - status is a small set.



HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=("route", "status"),
)

HTTP_REQUEST_LATENCY = Histogram(
    "http_request_latency_seconds",
    "HTTP request latency (seconds)",
    labelnames=("route",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10),
)

INFERENCE_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference latency (seconds)",
    labelnames=("provider",),
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
)
INFERENCE_LATENCY_SECONDS = INFERENCE_LATENCY

def render_metrics():
    data = generate_latest()
    return data, CONTENT_TYPE_LATEST
