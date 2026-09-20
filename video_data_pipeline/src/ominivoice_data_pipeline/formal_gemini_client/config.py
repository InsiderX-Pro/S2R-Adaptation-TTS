from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def default_credentials_path() -> Path:
    raw = (
        os.getenv("GEMINI_CREDENTIALS_JSON")
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    )
    if not raw:
        raise ValueError("set GEMINI_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS, or pass a credentials path")
    return Path(raw).expanduser().resolve()


@dataclass(frozen=True)
class GeminiConfig:
    credentials_path: Path | None = None
    model: str = "gemini-2.5-flash"
    location: str = "us-central1"
    timeout_s: float = 180.0
    retries: int = 1
    retry_base_s: float = 2.0
    proxy_url: str | None = None

    @classmethod
    def from_env(
        cls,
        *,
        credentials_path: str | Path | None = None,
        model: str | None = None,
        location: str | None = None,
        timeout_s: float | None = None,
        retries: int | None = None,
        retry_base_s: float | None = None,
        proxy_url: str | None = None,
    ) -> "GeminiConfig":
        env_proxy = os.getenv("GEMINI_PROXY_URL")
        return cls(
            credentials_path=Path(credentials_path or default_credentials_path()).expanduser().resolve(),
            model=str(model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip(),
            location=str(location or os.getenv("GEMINI_LOCATION") or "us-central1").strip(),
            timeout_s=float(timeout_s if timeout_s is not None else os.getenv("GEMINI_TIMEOUT_S", "180")),
            retries=max(1, int(retries if retries is not None else os.getenv("GEMINI_RETRIES", "1"))),
            retry_base_s=max(
                0.0,
                float(retry_base_s if retry_base_s is not None else os.getenv("GEMINI_RETRY_BASE_S", "2")),
            ),
            proxy_url=(proxy_url if proxy_url is not None else env_proxy) or None,
        )
