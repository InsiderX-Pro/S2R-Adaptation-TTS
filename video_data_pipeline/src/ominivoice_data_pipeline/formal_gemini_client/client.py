from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:  # The production extra supplies requests; keep offline config/tests importable.
    requests = None  # type: ignore[assignment]

from .config import GeminiConfig, default_credentials_path
from .proxy import ProxyProfile, discover_proxy_profiles


VERTEX_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class GeminiRequestError(RuntimeError):
    def __init__(self, message: str, *, retriable: bool = True) -> None:
        super().__init__(message)
        self.retriable = retriable


@dataclass(frozen=True)
class GeminiResponse:
    text: str
    raw_payload: dict[str, Any]
    model: str
    location: str
    proxy_profile: str
    attempt: int


def mime_type_for_path(path: Path) -> str:
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".m4a": "audio/mp4",
        ".aac": "audio/aac",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
    }.get(path.suffix.lower(), "application/octet-stream")


def extract_generate_content_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        feedback = payload.get("promptFeedback") or {}
        raise GeminiRequestError(f"Vertex returned no candidates; promptFeedback={feedback!r}", retriable=False)
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = [
        str(part.get("text"))
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str) and not part.get("thought")
    ]
    text = "\n".join(texts).strip()
    if not text:
        finish_reason = str((candidates[0] or {}).get("finishReason") or "unknown")
        raise GeminiRequestError(f"Vertex candidate contained no text; finishReason={finish_reason}")
    return text


class GeminiClient:
    """Vertex REST transport for the frozen Flash ASR request contract."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig.from_env()
        self.credentials_path = Path(self.config.credentials_path or default_credentials_path()).expanduser().resolve()
        if not self.credentials_path.is_file():
            raise FileNotFoundError(f"Gemini credentials not found: {self.credentials_path}")
        with self.credentials_path.open("r", encoding="utf-8") as handle:
            credential_metadata = json.load(handle)
        self.project_id = str(credential_metadata.get("project_id") or "").strip()
        if not self.project_id:
            raise ValueError("Gemini service-account JSON does not contain project_id.")

    @property
    def endpoint(self) -> str:
        return (
            f"https://{self.config.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}"
            f"/locations/{self.config.location}/publishers/google/models/{self.config.model}:generateContent"
        )

    @staticmethod
    def _session(profile: ProxyProfile) -> Any:
        if requests is None:
            raise RuntimeError("formal Gemini REST client requires requests")
        session = requests.Session()
        session.trust_env = False
        if profile.url:
            session.proxies.update(profile.requests_proxies)
        return session

    def _access_token(self, session: Any) -> str:
        try:
            from google.auth.exceptions import GoogleAuthError, RefreshError, TransportError
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
        except ImportError as exc:
            raise RuntimeError("formal Gemini REST client requires google-auth") from exc
        credentials = service_account.Credentials.from_service_account_file(
            str(self.credentials_path), scopes=[VERTEX_SCOPE]
        )
        try:
            credentials.refresh(Request(session=session))
        except TransportError as exc:
            raise GeminiRequestError("Google OAuth transport failed for this network profile.") from exc
        except RefreshError as exc:
            retriable = bool(getattr(exc, "retryable", False))
            message = "Google OAuth refresh failed temporarily." if retriable else "Google OAuth credential refresh was rejected."
            raise GeminiRequestError(message, retriable=retriable) from exc
        except GoogleAuthError as exc:
            raise GeminiRequestError("Google OAuth authentication failed.", retriable=False) from exc
        token = str(credentials.token or "").strip()
        if not token:
            raise GeminiRequestError("Google authentication returned an empty access token.")
        return token

    def _generate_once(self, request_payload: dict[str, Any], profile: ProxyProfile) -> dict[str, Any]:
        session = self._session(profile)
        token = self._access_token(session)
        response = session.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=request_payload,
            timeout=max(5.0, float(self.config.timeout_s)),
        )
        if response.status_code >= 400:
            body = (response.text or "").replace("\n", " ")[:500]
            retriable = response.status_code == 429 or response.status_code >= 500
            raise GeminiRequestError(
                f"Vertex generateContent returned HTTP {response.status_code}: {body}", retriable=retriable
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise GeminiRequestError("Vertex generateContent returned a non-JSON HTTP response.") from exc
        if not isinstance(payload, dict):
            raise GeminiRequestError("Vertex generateContent response root is not an object.")
        return payload

    def generate_content(
        self,
        prompt: str,
        *,
        media_path: str | Path | None = None,
        media_mime_type: str | None = None,
        temperature: float = 0.1,
        top_p: float | None = None,
        top_k: int | None = None,
        seed: int | None = None,
        thinking_budget: int | None = None,
        system_instruction: str | None = None,
        max_output_tokens: int | None = None,
        response_mime_type: str | None = None,
        response_schema: dict[str, Any] | None = None,
        retries: int | None = None,
    ) -> GeminiResponse:
        if not str(prompt or "").strip():
            raise ValueError("prompt cannot be empty")
        parts: list[dict[str, Any]] = []
        if media_path is not None:
            media = Path(media_path).expanduser().resolve()
            if not media.is_file():
                raise FileNotFoundError(f"Gemini media not found: {media}")
            parts.append(
                {"inlineData": {"mimeType": media_mime_type or mime_type_for_path(media), "data": base64.b64encode(media.read_bytes()).decode("ascii")}}
            )
        parts.append({"text": str(prompt)})

        generation_config: dict[str, Any] = {"temperature": float(temperature)}
        if top_p is not None:
            generation_config["topP"] = float(top_p)
        if top_k is not None:
            generation_config["topK"] = int(top_k)
        if seed is not None:
            generation_config["seed"] = int(seed)
        if thinking_budget is not None:
            generation_config["thinkingConfig"] = {"thinkingBudget": int(thinking_budget)}
        if max_output_tokens is not None:
            generation_config["maxOutputTokens"] = int(max_output_tokens)
        if response_mime_type:
            generation_config["responseMimeType"] = response_mime_type
        if response_schema:
            generation_config["responseSchema"] = response_schema
        request_payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation_config}
        if system_instruction and system_instruction.strip():
            request_payload["systemInstruction"] = {"parts": [{"text": system_instruction.strip()}]}

        profiles = discover_proxy_profiles(self.config.proxy_url)
        max_attempts = max(1, int(retries if retries is not None else self.config.retries))
        errors: list[str] = []
        for attempt in range(1, max_attempts + 1):
            for profile in profiles:
                try:
                    payload = self._generate_once(request_payload, profile)
                    return GeminiResponse(
                        text=extract_generate_content_text(payload), raw_payload=payload,
                        model=self.config.model, location=self.config.location,
                        proxy_profile=profile.name, attempt=attempt,
                    )
                except GeminiRequestError as exc:
                    errors.append(f"{profile.name}:{exc}")
                    if not exc.retriable:
                        raise
                except Exception as exc:
                    request_error = requests is not None and isinstance(exc, requests.RequestException)
                    if not request_error and not isinstance(exc, OSError):
                        raise
                    errors.append(f"{profile.name}:{type(exc).__name__}")
            if attempt < max_attempts and self.config.retry_base_s > 0:
                time.sleep(self.config.retry_base_s * (2 ** (attempt - 1)))
        detail = "; ".join(errors[-6:])
        raise GeminiRequestError(f"Vertex Gemini failed after {max_attempts} attempt(s): {detail}")


__all__ = ["GeminiClient", "GeminiRequestError", "GeminiResponse"]
