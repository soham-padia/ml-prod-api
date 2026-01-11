from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import Settings


def require_api_key(settings: Settings):
    async def _dep(x_api_key: str | None = Header(default=None, alias="X-API-Key")) -> None:
        if not settings.api_key:
            return  # disabled
        if not x_api_key or x_api_key != settings.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized",
            )

    return _dep
