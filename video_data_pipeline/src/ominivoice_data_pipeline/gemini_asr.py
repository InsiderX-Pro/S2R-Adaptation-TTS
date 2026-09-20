from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .formal_gemini_client import GeminiClient as FormalGeminiClient
from .formal_gemini_client import GeminiConfig as FormalGeminiConfig


PROMPT_ID = "p3_flash_canonical_greedy"
_PROMPT = """You are a deterministic blind ASR engine. The attached audio is the only evidence.
Return one single acoustically most likely verbatim transcription in the original language. Never translate, repair grammar, paraphrase, complete fragments, infer context, or offer alternatives. Preserve repetitions, fillers, incomplete words, names, and code-switching exactly as heard.
Use standard Unicode Myanmar spelling and one consistent lexical form. Do not vary synonyms or spelling merely because several forms are possible. Use Myanmar words for ordinary spoken numbers; retain ASCII digits only when the speaker clearly reads a code, identifier, or digit sequence. Keep audible English words in lowercase Latin letters. Use Myanmar punctuation only at clearly audible phrase boundaries and do not insert arbitrary spaces inside Myanmar words.
When audio is ambiguous, choose exactly one best acoustic hypothesis; do not list uncertainty and do not guess extra words. Set uncertain_spans to an empty array.
Return exactly one strict JSON object:
{"language":"detected language or language code","transcript":"single verbatim transcript","uncertain_spans":[]}"""
PROMPT_SHA256 = hashlib.sha256(_PROMPT.encode("utf-8")).hexdigest()

QUALITY_PROMPT_ID = "q3_flash_audio_text_voice_disguise_quality_guard"
_QUALITY_PROMPT = """You are a conservative audio-dataset quality auditor. The attached audio is the only evidence for acoustic quality. A frozen primary-ASR transcript is supplied below only for consistency checking.
Do not produce a new transcript, correction, alternative wording, or commentary. Do not judge grammar or topic. Judge each field independently and fail conservatively only when the evidence is audible.
speech_complete is false when either clip boundary cuts an audible word or syllable, or the speech is only an unfinished fragment. natural_short_response is true only for a very clear, self-contained, naturally spoken short reply such as an acknowledgement, yes/no answer, greeting, or discourse response. breath_or_noise_only is true when there is no usable lexical speech. Mark audible background music, sound effects, secondary/broadcast speech, low signal-to-noise ratio, clipping/distortion, and intentional voice disguise independently. voice_disguise_effect is true only when the target voice is clearly anonymized or disguised by strong pitch/formant shifting, robotic/electronic modulation, or similar identity-protection processing. Do not mark a naturally high/low voice, accent, ordinary EQ, compression, or codec artifacts as disguised. Treat conventional acronyms, initialisms, named entities, organization names, product names, technical terms, and short quoted English phrases as natural code-switching when audibly supported; examples include AA, NUG, ICC, DVB, US, UN, ASEAN, and official English institution names. english_excessive is true only when English materially dominates a clip that otherwise presents as Myanmar-language speech, not merely because one or several such terms occur. english_audio_text_consistent is false when English present in the supplied transcript is not supported by the audio or audible English is materially missing from it.
Return exactly one strict JSON object with a quality_assessment object and no transcript field.
"""
QUALITY_PROMPT_SHA256 = hashlib.sha256(_QUALITY_PROMPT.encode("utf-8")).hexdigest()

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "language": {"type": "STRING"},
        "transcript": {"type": "STRING"},
        "uncertain_spans": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["language", "transcript", "uncertain_spans"],
}

_QUALITY_FIELDS = (
    "speech_complete",
    "natural_short_response",
    "breath_or_noise_only",
    "background_music",
    "sound_effects",
    "broadcast_crosstalk",
    "voice_disguise_effect",
    "low_snr",
    "clipping",
    "english_natural_code_switch",
    "english_excessive",
    "english_audio_text_consistent",
)
QUALITY_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "quality_assessment": {
            "type": "OBJECT",
            "properties": {field: {"type": "BOOLEAN"} for field in _QUALITY_FIELDS},
            "required": list(_QUALITY_FIELDS),
        }
    },
    "required": ["quality_assessment"],
}


@dataclass(frozen=True)
class GeminiConfig:
    model: str = "gemini-2.5-flash"
    backend: str = "formal_rest"
    location: str = "us-central1"
    seeds: tuple[int, ...] = (101,)
    max_attempts: int = 1
    retry_wait_s: float = 2.0
    timeout_s: float = 180.0
    credentials_path: str | None = None
    proxy_url: str | None = None
    prompt_id: str = PROMPT_ID
    prompt: str = _PROMPT
    temperature: float = 0.0
    max_output_tokens: int = 1024
    thinking_budget: int = 0
    top_p: float | None = None
    top_k: int | None = 1
    quality_enabled: bool = True


def _extract_json(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Gemini response contains no JSON object")
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Gemini response JSON root must be an object")
    return value


def _parse_primary(payload: dict[str, Any]) -> dict[str, Any]:
    transcript = payload.get("transcript")
    uncertain = payload.get("uncertain_spans")
    if not isinstance(transcript, str):
        raise ValueError("Gemini transcript is not a string")
    if not isinstance(uncertain, list) or any(not isinstance(item, str) for item in uncertain):
        raise ValueError("Gemini uncertain_spans is not a string array")
    return {
        "language": str(payload.get("language") or "").strip(),
        "transcript": transcript.strip(),
        "uncertain_spans": uncertain,
    }


def _quality_assessment(payload: dict[str, Any]) -> dict[str, bool]:
    raw = payload.get("quality_assessment")
    if not isinstance(raw, dict):
        raise ValueError("Gemini response is missing quality_assessment")
    assessment: dict[str, bool] = {}
    for field in _QUALITY_FIELDS:
        value = raw.get(field)
        if not isinstance(value, bool):
            raise ValueError(f"Gemini quality_assessment.{field} must be boolean")
        assessment[field] = value
    return assessment


def _guard_normalized(text: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", text)
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def transcript_guard(transcript: str, duration_seconds: float) -> dict[str, Any]:
    normalized = _guard_normalized(transcript)
    duration = max(0.1, float(duration_seconds))
    char_limit = max(256, int(math.ceil(40.0 * duration)))
    fourgrams = Counter(normalized[index : index + 4] for index in range(max(0, len(normalized) - 3)))
    max_fourgram_count = max(fourgrams.values(), default=0)
    repetition_risk = len(normalized) > max(160, int(math.ceil(20.0 * duration))) and max_fourgram_count >= 12
    reasons: list[str] = []
    if len(normalized) > char_limit:
        reasons.append("transcript_too_long_for_audio")
    if repetition_risk:
        reasons.append("pathological_repetition")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "normalized_characters": len(normalized),
        "duration_seconds": duration_seconds,
        "characters_per_second": len(normalized) / duration,
        "character_limit": char_limit,
        "max_fourgram_count": max_fourgram_count,
    }


def _raw_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    first = candidates[0] if isinstance(candidates, list) and candidates and isinstance(candidates[0], dict) else {}
    return {
        "finish_reason": first.get("finishReason"),
        "usage_metadata": payload.get("usageMetadata") if isinstance(payload, dict) else None,
        "response_id": payload.get("responseId") if isinstance(payload, dict) else None,
        "model_version": payload.get("modelVersion") if isinstance(payload, dict) else None,
    }


class GeminiTranscriber:
    """Frozen P3 primary ASR plus an independent, non-transcribing quality observer."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig()
        self._client: FormalGeminiClient | None = None

    def load(self) -> None:
        if self._client is not None:
            return
        if self.config.backend.strip().lower() != "formal_rest":
            raise ValueError("production Gemini backend must be formal_rest")
        formal_config = FormalGeminiConfig.from_env(
            credentials_path=self.config.credentials_path,
            model=self.config.model,
            location=self.config.location,
            timeout_s=self.config.timeout_s,
            retries=1,
            proxy_url=self.config.proxy_url,
        )
        self._client = FormalGeminiClient(formal_config)

    def _primary_one(self, audio_path: Path, *, seed: int) -> dict[str, Any]:
        started = time.monotonic()
        last_error = ""
        for attempt in range(1, max(1, self.config.max_attempts) + 1):
            response: Any = None
            try:
                self.load()
                assert self._client is not None
                response = self._client.generate_content(
                    self.config.prompt,
                    media_path=str(audio_path),
                    media_mime_type="audio/wav",
                    temperature=float(self.config.temperature),
                    seed=int(seed),
                    max_output_tokens=int(self.config.max_output_tokens),
                    thinking_budget=int(self.config.thinking_budget),
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    response_mime_type="application/json",
                    response_schema=RESPONSE_SCHEMA,
                    retries=1,
                )
                raw_text = str(response.text or "").strip()
                parsed = _parse_primary(_extract_json(raw_text))
                with wave.open(str(audio_path), "rb") as handle:
                    duration_seconds = handle.getnframes() / max(1, handle.getframerate())
                guard = transcript_guard(parsed["transcript"], duration_seconds)
                return {
                    "status": "ok" if guard["passed"] else "invalid_response",
                    "seed": int(seed),
                    **parsed,
                    "guard": guard,
                    "raw_response_text": raw_text,
                    "model": self.config.model,
                    "backend": "formal_rest",
                    "client_contract": "gemini_asr_single_seed_v1.vendor.client",
                    "prompt_id": self.config.prompt_id,
                    "prompt_sha256": PROMPT_SHA256,
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                    "thinking_budget": self.config.thinking_budget,
                    "top_p": self.config.top_p,
                    "top_k": self.config.top_k,
                    "proxy_url": self.config.proxy_url,
                    "response_model": response.model,
                    "response_location": response.location,
                    "proxy_profile": response.proxy_profile,
                    "client_attempt": response.attempt,
                    **_raw_metadata(response.raw_payload),
                    "attempt": attempt,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < max(1, self.config.max_attempts):
                    time.sleep(max(0.0, self.config.retry_wait_s) * attempt)
        return {
            "status": "error",
            "seed": int(seed),
            "transcript": "",
            "model": self.config.model,
            "backend": "formal_rest",
            "client_contract": "gemini_asr_single_seed_v1.vendor.client",
            "prompt_id": self.config.prompt_id,
            "prompt_sha256": PROMPT_SHA256,
            "top_k": self.config.top_k,
            "raw_response_text": str(response.text or "").strip() if response is not None else "",
            **(_raw_metadata(response.raw_payload) if response is not None else {}),
            "error": last_error,
            "elapsed_s": round(time.monotonic() - started, 3),
        }

    def transcribe_clip(self, audio_path: str | Path, *, seeds: Iterable[int] | None = None) -> dict[str, Any]:
        source = Path(audio_path).expanduser().resolve()
        selected = tuple(int(seed) for seed in (seeds if seeds is not None else self.config.seeds))
        if not selected:
            raise ValueError("at least one Gemini seed is required")
        runs = [self._primary_one(source, seed=seed) for seed in selected]
        return {
            "audio_path": str(source),
            "model": self.config.model,
            "backend": "formal_rest",
            "client_contract": "gemini_asr_single_seed_v1.vendor.client",
            "prompt_id": self.config.prompt_id,
            "prompt_sha256": PROMPT_SHA256,
            "decoding": {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
                "thinking_budget": self.config.thinking_budget,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "seed_policy": "single_seed",
            },
            "runs": runs,
            "success_count": sum(1 for row in runs if row.get("status") == "ok"),
            "requested_count": len(runs),
        }

    def assess_clip(self, audio_path: str | Path, transcript: str, *, seed: int = 101) -> dict[str, Any]:
        source = Path(audio_path).expanduser().resolve()
        request_prompt = _QUALITY_PROMPT + "\nFrozen primary-ASR transcript JSON:\n" + json.dumps(
            {"transcript": str(transcript or "")}, ensure_ascii=False, separators=(",", ":")
        )
        request_sha256 = hashlib.sha256(request_prompt.encode("utf-8")).hexdigest()
        started = time.monotonic()
        try:
            self.load()
            assert self._client is not None
            response = self._client.generate_content(
                request_prompt,
                media_path=str(source),
                media_mime_type="audio/wav",
                temperature=0.0,
                seed=int(seed),
                max_output_tokens=512,
                thinking_budget=0,
                top_p=None,
                top_k=1,
                response_mime_type="application/json",
                response_schema=QUALITY_RESPONSE_SCHEMA,
                retries=1,
            )
            raw_text = str(response.text or "").strip()
            assessment = _quality_assessment(_extract_json(raw_text))
            return {
                "status": "ok",
                "seed": int(seed),
                "quality_assessment": assessment,
                "raw_response_text": raw_text,
                "model": self.config.model,
                "backend": "formal_rest",
                "client_contract": "gemini_asr_single_seed_v1.vendor.client",
                "prompt_id": QUALITY_PROMPT_ID,
                "prompt_sha256": QUALITY_PROMPT_SHA256,
                "request_prompt_sha256": request_sha256,
                "primary_transcript_sha256": hashlib.sha256(str(transcript or "").encode("utf-8")).hexdigest(),
                "temperature": 0.0,
                "thinking_budget": 0,
                "top_p": None,
                "top_k": 1,
                "max_output_tokens": 512,
                "response_model": response.model,
                "response_location": response.location,
                "proxy_profile": response.proxy_profile,
                "client_attempt": response.attempt,
                **_raw_metadata(response.raw_payload),
                "elapsed_s": round(time.monotonic() - started, 3),
            }
        except Exception as exc:
            return {
                "status": "error",
                "seed": int(seed),
                "quality_assessment": {},
                "prompt_id": QUALITY_PROMPT_ID,
                "prompt_sha256": QUALITY_PROMPT_SHA256,
                "request_prompt_sha256": request_sha256,
                "primary_transcript_sha256": hashlib.sha256(str(transcript or "").encode("utf-8")).hexdigest(),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_s": round(time.monotonic() - started, 3),
            }


__all__ = [
    "GeminiConfig",
    "GeminiTranscriber",
    "PROMPT_ID",
    "PROMPT_SHA256",
    "QUALITY_PROMPT_ID",
    "QUALITY_PROMPT_SHA256",
    "RESPONSE_SCHEMA",
    "QUALITY_RESPONSE_SCHEMA",
    "transcript_guard",
]
