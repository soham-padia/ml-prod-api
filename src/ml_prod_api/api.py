import time
from typing import Any

from fastapi import Body, Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from . import __version__
from .auth import require_api_key
from .config import Settings, get_settings
from .logging import configure_logging, get_logger
from .middleware import RequestContextMiddleware
from .metrics import INFERENCE_LATENCY_SECONDS, render_metrics
from .runtime import load_model
from .schemas import (
    HealthResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    Prediction,
    RootResponse,
)

log = get_logger(__name__)
limiter = Limiter(key_func=get_remote_address)


def create_app(settings: Settings) -> FastAPI:
    configure_logging(settings.log_level)

    app = FastAPI(title="ml-prod-api", version=__version__)

    # Rate limiting middleware (optional via effective limit)
    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    effective_limit = (
        settings.rate_limit_default
        if settings.rate_limit_enabled
        else "1000000/minute"
    )

    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return JSONResponse(status_code=429, content={"detail": "Too Many Requests"})

    # Request context + structured logging
    app.add_middleware(RequestContextMiddleware, settings=settings)

    # CORS (optional)
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["*"],
        )

    # Load model at startup
    app.state.model = None

    @app.on_event("startup")
    async def _startup() -> None:
        try:
            app.state.model = load_model(settings.model_provider, settings.artifact_dir)
            log.info(
                "model loaded",
                extra={"model_version": settings.model_version, "route": "startup"},
            )
        except Exception:
            # Keep process alive; readiness will fail.
            log.error(
                "model failed to load",
                extra={"model_version": settings.model_version, "route": "startup"},
            )
            app.state.model = None

    def _validate_limits(text: str) -> None:
        if len(text) > settings.max_text_length:
            raise ValueError(f"text too long (max {settings.max_text_length})")

    def _label(prob_toxic: float) -> str:
        return "toxic" if prob_toxic >= 0.5 else "non_toxic"

    # Root
    @app.get("/", response_model=RootResponse)
    async def root() -> RootResponse:
        return RootResponse(
            name="ml-prod-api",
            version=__version__,
            model_version=settings.model_version,
            provider=settings.model_provider,
        )

    # Health (liveness)
    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    # Ready (readiness)
    @app.get("/readyz", response_model=HealthResponse)
    async def readyz() -> Response:
        if app.state.model is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return JSONResponse(status_code=200, content={"status": "ready"})

    # Metrics
    @app.get("/metrics")
    async def metrics() -> Response:
        data, content_type = render_metrics()
        return Response(content=data, media_type=content_type)

    # Predict (single)
    @app.post("/v1/predict", response_model=PredictResponse)
    @limiter.limit(effective_limit)
    async def predict(
        request: Request,
        req: PredictRequest = Body(...),
        _: Any = Depends(require_api_key(settings)),
    ) -> PredictResponse:
        if app.state.model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not ready"})

        _validate_limits(req.text)

        # Record inference latency in Prometheus histogram
        # Record inference latency in Prometheus histogram
        with INFERENCE_LATENCY_SECONDS.labels(provider=settings.model_provider).time():
            pred = app.state.model.predict_one(req.text)


        p = float(pred.probability_toxic)
        out = Prediction(
            label=_label(p),
            probability_toxic=p,
            model_version=settings.model_version,
            provider=settings.model_provider,
        )
        return PredictResponse(prediction=out)

    # Predict (batch)
    @app.post("/v1/predict_batch", response_model=PredictBatchResponse)
    @limiter.limit(effective_limit)
    async def predict_batch(
        request: Request,
        req: PredictBatchRequest = Body(...),
        _: Any = Depends(require_api_key(settings)),
    ) -> PredictBatchResponse:
        if app.state.model is None:
            return JSONResponse(status_code=503, content={"detail": "Model not ready"})

        if len(req.texts) > settings.max_batch_size:
            return JSONResponse(
                status_code=422,
                content={"detail": f"batch too large (max {settings.max_batch_size})"},
            )

        for t in req.texts:
            _validate_limits(t)

        # Record inference latency in Prometheus histogram
        with INFERENCE_LATENCY_SECONDS.labels(provider=settings.model_provider).time():
            preds = app.state.model.predict_batch(req.texts)


        outs = []
        for pred in preds:
            p = float(pred.probability_toxic)
            outs.append(
                Prediction(
                    label=_label(p),
                    probability_toxic=p,
                    model_version=settings.model_version,
                    provider=settings.model_provider,
                )
            )

        return PredictBatchResponse(predictions=outs)

    # Error normalization for validation limit errors
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        log.error(
            "unhandled exception",
            extra={
                "request_id": getattr(request.state, "request_id", None),
                "route": request.url.path,
                "status_code": 500,
                "latency_ms": None,
                "model_version": settings.model_version,
            },
        )
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})

    return app


settings = get_settings()
app = create_app(settings)
