#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
CONCURRENCY="${CONCURRENCY:-20}"
REQUESTS="${REQUESTS:-2000}"

echo "Warming up..."
curl -s "${BASE_URL}/readyz" >/dev/null || true

echo "Benchmark: POST /v1/predict"
hey -n "${REQUESTS}" -c "${CONCURRENCY}" \
  -m POST \
  -H "Content-Type: application/json" \
  -D <(printf '{"text":"you are awful and I hate you"}') \
  "${BASE_URL}/v1/predict"

echo
echo "Benchmark: POST /v1/predict_batch"
hey -n "${REQUESTS}" -c "${CONCURRENCY}" \
  -m POST \
  -H "Content-Type: application/json" \
  -D <(printf '{"texts":["you are awful","hello friend","go away","have a nice day","you are stupid"]}') \
  "${BASE_URL}/v1/predict_batch"
