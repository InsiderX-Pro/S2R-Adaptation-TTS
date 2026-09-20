from __future__ import annotations

import math
import re
import unicodedata
import wave
from array import array
from pathlib import Path
from typing import Any, Mapping, Sequence

from .language_profiles import get_language_profile, script_code


_ALLOWED_FORMAT_CHARACTERS = {"\u200b", "\u200c", "\u200d"}
_ALLOWED_SYMBOL_CHARACTERS = set("%+&@#=/$-*°")
_WHITESPACE_RE = re.compile(r"\s+")

_DEFAULT_TEXT_CONFIG: dict[str, Any] = {
    "target_language": "my",
    "min_target_script_letters": 0,
    "min_target_script_ratio": 0.0,
    "max_latin_ratio": 0.65,
    "min_unnatural_latin_ratio": 0.35,
    "max_repeated_token_run": 3,
    "max_repeated_fourgram_count": 11,
}

_DEFAULT_AUDIO_CONFIG: dict[str, Any] = {
    "frame_ms": 30.0,
    "clipping_amplitude_ratio": 0.999,
    "max_clipped_sample_ratio": 0.01,
    "max_clipped_run_ms": 3.0,
    "snr_noise_quantile": 0.20,
    "snr_signal_quantile": 0.90,
    "min_snr_db": 8.0,
    "min_snr_duration_ms": 1_000,
}


def _config(defaults: Mapping[str, Any], value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    result = dict(defaults)
    if value is None:
        return result
    unknown = sorted(set(value) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {name} keys: {unknown}")
    result.update(value)
    return result


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFC", str(text or ""))
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _is_myanmar(character: str) -> bool:
    return get_language_profile("my").contains(character)


def _is_latin(character: str) -> bool:
    return "LATIN" in unicodedata.name(character, "")


def _abnormal_characters(text: str, *, target_language: str) -> list[dict[str, str]]:
    target = get_language_profile(target_language)
    abnormal: list[dict[str, str]] = []
    for character in text:
        if character.isspace():
            continue
        category = unicodedata.category(character)
        if character == "\ufffd" or category in {"Cc", "Cs", "Co", "Cn"}:
            abnormal.append(
                {
                    "character": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "reason": "invalid_unicode_category",
                }
            )
            continue
        if category == "Cf" and character not in _ALLOWED_FORMAT_CHARACTERS:
            abnormal.append(
                {
                    "character": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "reason": "unexpected_format_character",
                }
            )
            continue
        if category.startswith("S") and character not in _ALLOWED_SYMBOL_CHARACTERS:
            abnormal.append(
                {
                    "character": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "reason": "unexpected_symbol",
                }
            )
            continue
        if category.startswith("L") and not (target.contains(character) or _is_latin(character)):
            abnormal.append(
                {
                    "character": character,
                    "codepoint": f"U+{ord(character):04X}",
                    "reason": "unexpected_letter_script",
                }
            )
    return abnormal


def _maximum_token_run(tokens: Sequence[str]) -> int:
    longest = 0
    current = 0
    previous = ""
    for token in tokens:
        if token == previous:
            current += 1
        else:
            previous = token
            current = 1
        longest = max(longest, current)
    return longest


def _maximum_fourgram_count(text: str) -> int:
    compact = "".join(
        character
        for character in text
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )
    if len(compact) < 4:
        return 0
    counts: dict[str, int] = {}
    for index in range(len(compact) - 3):
        gram = compact[index : index + 4]
        counts[gram] = counts.get(gram, 0) + 1
    return max(counts.values(), default=0)


def text_quality_report(text: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = _config(_DEFAULT_TEXT_CONFIG, config, name="text quality config")
    target_language = get_language_profile(str(selected["target_language"])).code
    min_target_script_letters = int(selected["min_target_script_letters"])
    min_target_script_ratio = float(selected["min_target_script_ratio"])
    if min_target_script_letters < 0:
        raise ValueError("text_quality.min_target_script_letters must be non-negative")
    if not 0.0 <= min_target_script_ratio <= 1.0:
        raise ValueError("text_quality.min_target_script_ratio must be between 0 and 1")
    max_latin_ratio = float(selected["max_latin_ratio"])
    if not 0.0 <= max_latin_ratio <= 1.0:
        raise ValueError("text_quality.max_latin_ratio must be between 0 and 1")
    min_unnatural_latin_ratio = float(selected["min_unnatural_latin_ratio"])
    if not 0.0 <= min_unnatural_latin_ratio <= max_latin_ratio:
        raise ValueError(
            "text_quality.min_unnatural_latin_ratio must be between 0 and max_latin_ratio"
        )
    max_token_run = int(selected["max_repeated_token_run"])
    max_fourgram = int(selected["max_repeated_fourgram_count"])
    if max_token_run < 1 or max_fourgram < 1:
        raise ValueError("text repetition thresholds must be positive")

    normalized = normalize_transcript(text)
    abnormal = _abnormal_characters(normalized, target_language=target_language)
    letters = [character for character in normalized if unicodedata.category(character).startswith("L")]
    latin_letters = sum(_is_latin(character) for character in letters)
    myanmar_letters = sum(_is_myanmar(character) for character in letters)
    target = get_language_profile(target_language)
    target_script_letters = sum(target.contains(character) for character in letters)
    other_script_letters = sum(
        script_code(character) not in {None, "latin", target_language}
        for character in letters
    )
    latin_ratio = latin_letters / len(letters) if letters else 0.0
    target_script_ratio = target_script_letters / len(letters) if letters else 0.0
    tokens = [token.casefold() for token in normalized.split() if token]
    maximum_token_run = _maximum_token_run(tokens)
    maximum_fourgram_count = _maximum_fourgram_count(normalized)

    reasons: list[str] = []
    if not normalized:
        reasons.append("empty_text")
    if abnormal:
        reasons.append("abnormal_characters")
    if maximum_token_run > max_token_run or maximum_fourgram_count > max_fourgram:
        reasons.append("repetitive_text")
    if latin_ratio > max_latin_ratio:
        reasons.append("excessive_english")
    if (
        target_script_letters < min_target_script_letters
        or target_script_ratio < min_target_script_ratio
    ):
        reasons.append("insufficient_target_script")
    return {
        "schema_version": "ominivoice.text-quality.v1",
        "normalized_text": normalized,
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "target_language": target_language,
            "character_count": len(normalized),
            "letter_count": len(letters),
            "latin_letter_count": latin_letters,
            "myanmar_letter_count": myanmar_letters,
            "target_script_letter_count": target_script_letters,
            "other_supported_script_letter_count": other_script_letters,
            "latin_ratio": round(latin_ratio, 6),
            "target_script_ratio": round(target_script_ratio, 6),
            "maximum_repeated_token_run": maximum_token_run,
            "maximum_repeated_fourgram_count": maximum_fourgram_count,
        },
        "abnormal_characters": abnormal,
    }


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("-inf")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _decode_pcm(fragment: bytes, sample_width: int) -> list[int]:
    if sample_width == 1:
        return [value - 128 for value in fragment]
    if sample_width == 2:
        samples = array("h")
        samples.frombytes(fragment)
        if samples.itemsize != 2:
            raise RuntimeError("unexpected native 16-bit sample size")
        return list(samples)
    if sample_width == 3:
        decoded: list[int] = []
        for index in range(0, len(fragment) - 2, 3):
            value = int.from_bytes(fragment[index : index + 3], byteorder="little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            decoded.append(value)
        return decoded
    if sample_width == 4:
        samples = array("i")
        samples.frombytes(fragment)
        if samples.itemsize != 4:
            raise RuntimeError("unexpected native 32-bit sample size")
        return list(samples)
    raise ValueError(f"unsupported PCM sample width: {sample_width}")


def analyze_wav_quality(path: str | Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = _config(_DEFAULT_AUDIO_CONFIG, config, name="audio quality config")
    frame_ms = float(selected["frame_ms"])
    clipping_amplitude_ratio = float(selected["clipping_amplitude_ratio"])
    max_clipped_sample_ratio = float(selected["max_clipped_sample_ratio"])
    max_clipped_run_ms = float(selected["max_clipped_run_ms"])
    noise_quantile = float(selected["snr_noise_quantile"])
    signal_quantile = float(selected["snr_signal_quantile"])
    min_snr_db = float(selected["min_snr_db"])
    min_snr_duration_ms = int(selected["min_snr_duration_ms"])
    if frame_ms <= 0.0:
        raise ValueError("audio_quality.frame_ms must be positive")
    if not 0.0 < clipping_amplitude_ratio <= 1.0:
        raise ValueError("audio_quality.clipping_amplitude_ratio must be in (0, 1]")
    if not 0.0 <= max_clipped_sample_ratio <= 1.0:
        raise ValueError("audio_quality.max_clipped_sample_ratio must be between 0 and 1")
    if not 0.0 <= noise_quantile < signal_quantile <= 1.0:
        raise ValueError("audio quality SNR quantiles are invalid")

    source = Path(path).expanduser().resolve()
    with wave.open(str(source), "rb") as handle:
        if handle.getcomptype() != "NONE":
            raise ValueError("audio quality analysis requires uncompressed PCM WAV")
        channels = handle.getnchannels()
        sample_rate = handle.getframerate()
        sample_width = handle.getsampwidth()
        frame_count = handle.getnframes()
        fragment = handle.readframes(frame_count)
    samples = _decode_pcm(fragment, sample_width)
    if channels > 1:
        samples = [
            round(sum(samples[index : index + channels]) / channels)
            for index in range(0, len(samples), channels)
            if len(samples[index : index + channels]) == channels
        ]
    peak_value = float((1 << (sample_width * 8 - 1)) - 1)
    clip_threshold = peak_value * clipping_amplitude_ratio
    clipped_count = 0
    longest_clipped_run = 0
    current_clipped_run = 0
    for sample in samples:
        if abs(sample) >= clip_threshold:
            clipped_count += 1
            current_clipped_run += 1
            longest_clipped_run = max(longest_clipped_run, current_clipped_run)
        else:
            current_clipped_run = 0
    clipped_sample_ratio = clipped_count / len(samples) if samples else 0.0
    longest_clipped_run_ms = longest_clipped_run * 1000.0 / max(1, sample_rate)

    samples_per_frame = max(1, round(sample_rate * frame_ms / 1000.0))
    frame_levels: list[float] = []
    for index in range(0, len(samples), samples_per_frame):
        frame = samples[index : index + samples_per_frame]
        if not frame:
            continue
        rms = math.sqrt(sum(float(sample) * float(sample) for sample in frame) / len(frame))
        frame_levels.append(20.0 * math.log10(max(rms / peak_value, 1e-9)))
    if frame_levels:
        noise_floor_dbfs = _quantile(frame_levels, noise_quantile)
        signal_level_dbfs = _quantile(frame_levels, signal_quantile)
        estimated_snr_db = signal_level_dbfs - noise_floor_dbfs
    else:
        noise_floor_dbfs = -180.0
        signal_level_dbfs = -180.0
        estimated_snr_db = 0.0
    duration_ms = round(len(samples) * 1000.0 / max(1, sample_rate))

    reasons: list[str] = []
    if clipped_sample_ratio > max_clipped_sample_ratio or longest_clipped_run_ms > max_clipped_run_ms:
        reasons.append("clipping_detected")
    if duration_ms >= min_snr_duration_ms and estimated_snr_db < min_snr_db:
        reasons.append("low_snr_detected")
    return {
        "schema_version": "ominivoice.audio-quality.v1",
        "audio_path": str(source),
        "passed": not reasons,
        "reasons": reasons,
        "metrics": {
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "channels": channels,
            "sample_width_bytes": sample_width,
            "clipped_sample_count": clipped_count,
            "clipped_sample_ratio": round(clipped_sample_ratio, 8),
            "longest_clipped_run_ms": round(longest_clipped_run_ms, 3),
            "noise_floor_dbfs": round(noise_floor_dbfs, 3),
            "signal_level_dbfs": round(signal_level_dbfs, 3),
            "estimated_snr_db": round(estimated_snr_db, 3),
        },
    }


__all__ = ["analyze_wav_quality", "normalize_transcript", "text_quality_report"]
