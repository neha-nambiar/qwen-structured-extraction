"""Settings from the environment or .env, validated by pydantic."""

import logging
from typing import Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

log = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", hide_input_in_errors=True, extra="ignore")

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    database_url: SecretStr | None = None
    redis_url: SecretStr | None = None

    api_key: SecretStr | None = None

    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    # RTX 4090, 4829 records: 8 -> 0.373 s/record, 16 -> 0.256, 32 -> 0.252.
    # Flat past 16; accuracy unchanged.
    max_concurrent_requests: int = Field(default=32, ge=1, le=256)
    request_timeout_seconds: float = Field(default=120.0, gt=0)
    # Unset means no record is flagged on confidence.
    confidence_threshold: float | None = Field(default=None, le=0)

    max_upload_bytes: int = Field(default=209_715_200, gt=0)  # 200 MiB
    hourly_usd: float = Field(default=0.74, gt=0)

    runpod_pod_id: str | None = None
    runpod_api_key: SecretStr | None = None


def load_settings() -> Settings:
    """Build Settings and configure logging; exit 2 on invalid settings."""
    # Log setting names and error types only, never values.
    try:
        settings = Settings()
    except ValidationError as exc:
        logging.basicConfig()
        for error in exc.errors():
            log.error(
                "invalid setting %s: %s",
                ".".join(str(part) for part in error["loc"]),
                error["type"],
            )
        raise SystemExit(2) from None
    logging.basicConfig(level=settings.log_level)
    return settings


def require(settings: Settings, *names: str) -> None:
    """Exit 2 if any named setting is unset or empty."""
    missing = []
    for name in names:
        value = getattr(settings, name)
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if value is None or value == "":
            missing.append(name.upper())
    if missing:
        log.error("missing required settings: %s", ", ".join(missing))
        raise SystemExit(2)
