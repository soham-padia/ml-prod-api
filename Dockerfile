# syntax=docker/dockerfile:1

########################
# Builder
########################
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt pyproject.toml ruff.toml /build/
RUN pip install --upgrade pip && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

########################
# Runtime
########################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOST=0.0.0.0 \
    APP_PORT=8000

WORKDIR /app

# Create non-root user
RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /wheels /wheels
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r /app/requirements.txt \
  && rm -rf /wheels

COPY src /app/src
COPY artifacts /app/artifacts

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz').read()"

CMD ["uvicorn", "ml_prod_api.api:app", "--host", "0.0.0.0", "--port", "8000"]
