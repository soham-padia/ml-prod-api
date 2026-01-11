from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import Settings
from .logging import get_logger
from .metrics import HTTP_REQUESTS_TOTAL, HTTP_REQUEST_LATENCY

log = get_logger(__name__)


def _route_template(request: Request) -> str:
    # Best-effort templated route
    route = request.scope.get("route")
    if route and hasattr(route, "path"):
        return str(route.path)
    return request.url.path


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings

    async def dispatch(self, request: Request, call_next: Callable):
        request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500
        route = _route_template(request)

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_s = max(0.0, time.perf_counter() - start)
            latency_ms = int(latency_s * 1000)

            # Metrics
            if self.settings.metrics_enabled:
                HTTP_REQUESTS_TOTAL.labels(route=route, status=str(status_code)).inc()
                HTTP_REQUEST_LATENCY.labels(route=route).observe(latency_s)

            # Structured log (no raw text)
            log.info(
                "request completed",
                extra={
                    "request_id": request_id,
                    "route": route,
                    "status_code": status_code,
                    "latency_ms": latency_ms,
                    "model_version": self.settings.model_version,
                },
            )
