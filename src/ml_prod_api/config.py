from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=None,
        extra="ignore",
        protected_namespaces=(),  # allow fields like model_provider/model_version
    )


    # App
    app_env: str = "local"
    log_level: str = "INFO"

    # Model loading
    model_provider: str = "stub"  # baseline | transformer | stub
    model_version: str = "dev"
    artifact_dir: str = "artifacts/dev"

    # Validation limits
    max_text_length: int = 512
    max_batch_size: int = 64

    # Security
    cors_allow_origins: str = ""  # comma-separated; empty => no CORS middleware
    api_key: str = ""  # if set, require X-API-Key header

    # Rate limit
    rate_limit_enabled: bool = True
    rate_limit_default: str = "60/minute"

    # Metrics
    metrics_enabled: bool = True


def get_settings() -> Settings:
    return Settings()
