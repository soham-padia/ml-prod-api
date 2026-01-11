from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s %(request_id)s %(route)s "
        "%(status_code)s %(latency_ms)s %(model_version)s"
    )
    handler.setFormatter(formatter)

    # Replace existing handlers to avoid duplicate logs (uvicorn reload etc.)
    root.handlers = [handler]


class SafeExtraAdapter(logging.LoggerAdapter):
    """
    Ensures extra keys exist for structured logs.
    """

    def process(self, msg: str, kwargs: dict[str, Any]):
        extra = kwargs.get("extra", {})
        extra.setdefault("request_id", None)
        extra.setdefault("route", None)
        extra.setdefault("status_code", None)
        extra.setdefault("latency_ms", None)
        extra.setdefault("model_version", None)
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(name: str) -> SafeExtraAdapter:
    return SafeExtraAdapter(logging.getLogger(name), {})
