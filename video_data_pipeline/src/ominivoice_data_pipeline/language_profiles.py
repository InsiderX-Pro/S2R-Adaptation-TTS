from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class LanguageProfile:
    code: str
    english_name: str
    native_name: str
    unicode_ranges: tuple[tuple[int, int], ...]
    youtube_subtitle_codes: tuple[str, ...]
    primary_instruction: str

    def contains(self, character: str) -> bool:
        codepoint = ord(character)
        return any(start <= codepoint <= end for start, end in self.unicode_ranges)


@dataclass(frozen=True)
class PromptContract:
    language: LanguageProfile
    primary_id: str
    primary_prompt: str
    primary_sha256: str
    quality_id: str
    quality_prompt: str
    quality_sha256: str


LANGUAGE_PROFILES: dict[str, LanguageProfile] = {
    "my": LanguageProfile(
        code="my",
        english_name="Myanmar",
        native_name="မြန်မာ",
        unicode_ranges=((0x1000, 0x109F), (0xA9E0, 0xA9FF), (0xAA60, 0xAA7F)),
        youtube_subtitle_codes=("my-orig", "my"),
        primary_instruction=(
            "Use standard Unicode Myanmar spelling and one consistent lexical form. Do not vary synonyms or "
            "spelling merely because several forms are possible. Use Myanmar words for ordinary spoken numbers; "
            "retain ASCII digits only when the speaker clearly reads a code, identifier, or digit sequence. Keep "
            "audible English words in lowercase Latin letters. Use Myanmar punctuation only at clearly audible "
            "phrase boundaries and do not insert arbitrary spaces inside Myanmar words."
        ),
    ),
    "lo": LanguageProfile(
        code="lo",
        english_name="Lao",
        native_name="ລາວ",
        unicode_ranges=((0x0E80, 0x0EFF),),
        youtube_subtitle_codes=("lo-orig", "lo"),
        primary_instruction=(
            "Use standard Unicode Lao spelling and one consistent lexical form. Do not vary synonyms or spelling "
            "merely because several forms are possible. Use Lao words for ordinary spoken numbers; retain ASCII "
            "digits only when the speaker clearly reads a code, identifier, or digit sequence. Keep audible English "
            "words in lowercase Latin letters. Follow normal Lao punctuation and spacing conventions and do not "
            "insert arbitrary spaces inside Lao words."
        ),
    ),
    "km": LanguageProfile(
        code="km",
        english_name="Khmer",
        native_name="ខ្មែរ",
        unicode_ranges=((0x1780, 0x17FF), (0x19E0, 0x19FF)),
        youtube_subtitle_codes=("km-orig", "km"),
        primary_instruction=(
            "Use standard Unicode Khmer spelling and one consistent lexical form. Do not vary synonyms or spelling "
            "merely because several forms are possible. Use Khmer words for ordinary spoken numbers; retain ASCII "
            "digits only when the speaker clearly reads a code, identifier, or digit sequence. Keep audible English "
            "words in lowercase Latin letters. Follow normal Khmer punctuation and spacing conventions, including "
            "natural word-boundary spaces, without inventing spaces that are not justified by the utterance."
        ),
    ),
}


def get_language_profile(code: str) -> LanguageProfile:
    normalized = str(code or "").strip().lower().replace("_", "-")
    aliases = {
        "burmese": "my",
        "myanmar": "my",
        "lao": "lo",
        "khmer": "km",
        "cambodian": "km",
    }
    normalized = aliases.get(normalized, normalized)
    try:
        return LANGUAGE_PROFILES[normalized]
    except KeyError as exc:
        raise ValueError(f"unsupported target language: {code!r}; choose from {sorted(LANGUAGE_PROFILES)}") from exc


def script_code(character: str) -> str | None:
    for code, profile in LANGUAGE_PROFILES.items():
        if profile.contains(character):
            return code
    if "LATIN" in unicodedata.name(character, ""):
        return "latin"
    return None


def script_metrics(text: str, target_language: str) -> dict[str, float | int | str]:
    target = get_language_profile(target_language)
    counts = {code: 0 for code in LANGUAGE_PROFILES}
    counts["latin"] = 0
    other_letters = 0
    for character in unicodedata.normalize("NFC", str(text or "")):
        if not unicodedata.category(character).startswith(("L", "M")):
            continue
        code = script_code(character)
        if code is None:
            if unicodedata.category(character).startswith("L"):
                other_letters += 1
            continue
        counts[code] += 1
    considered = sum(counts.values()) + other_letters
    target_count = counts[target.code]
    non_latin = considered - counts["latin"]
    return {
        "target_language": target.code,
        "target_script_letters": target_count,
        "latin_letters": counts["latin"],
        "other_supported_script_letters": sum(
            value for code, value in counts.items() if code not in {target.code, "latin"}
        ),
        "other_letters": other_letters,
        "considered_letters": considered,
        "target_script_ratio": (target_count / considered if considered else 0.0),
        "target_non_latin_ratio": (target_count / non_latin if non_latin else 0.0),
    }


def normalized_title_key(title: str) -> str:
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFC", str(title or ""))
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def _replace_primary_language_instruction(base_prompt: str, profile: LanguageProfile) -> str:
    lines = str(base_prompt).splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("Use standard Unicode Myanmar spelling"):
            lines[index] = profile.primary_instruction
            replaced = True
            break
    if not replaced:
        raise ValueError("base ASR prompt does not contain the expected Myanmar language instruction")
    return "\n".join(lines)


def _replace_quality_language(base_prompt: str, profile: LanguageProfile) -> str:
    prompt = str(base_prompt).replace("Myanmar-language", f"{profile.english_name}-language")
    prompt = prompt.replace(
        "examples include AA, NUG, ICC, DVB, US, UN, ASEAN, and official English institution names",
        "examples include UN, ASEAN, AI, US, EU, and official English institution names",
    )
    if profile.code != "my" and "Myanmar-language" in prompt:
        raise ValueError("quality prompt still contains a Myanmar-only language reference")
    return prompt


def build_prompt_contract(
    language: str,
    *,
    base_primary_id: str,
    base_primary_prompt: str,
    base_quality_id: str,
    base_quality_prompt: str,
) -> PromptContract:
    profile = get_language_profile(language)
    if profile.code == "my":
        primary_prompt = str(base_primary_prompt)
        quality_prompt = str(base_quality_prompt)
        primary_id = str(base_primary_id)
        quality_id = str(base_quality_id)
    else:
        primary_prompt = _replace_primary_language_instruction(base_primary_prompt, profile)
        quality_prompt = _replace_quality_language(base_quality_prompt, profile)
        primary_id = f"{base_primary_id}_{profile.code}_v1"
        quality_id = f"{base_quality_id}_{profile.code}_v1"
    return PromptContract(
        language=profile,
        primary_id=primary_id,
        primary_prompt=primary_prompt,
        primary_sha256=hashlib.sha256(primary_prompt.encode("utf-8")).hexdigest(),
        quality_id=quality_id,
        quality_prompt=quality_prompt,
        quality_sha256=hashlib.sha256(quality_prompt.encode("utf-8")).hexdigest(),
    )


def activate_pipeline_language(language: str) -> PromptContract:
    """Activate one language contract for the lifetime of the current process.

    The production pipeline historically imports prompt constants by value.  A
    dedicated process per language lets us update those provenance constants
    without changing the validated Myanmar default or mixing contracts in one
    run.
    """

    from . import gemini_asr, pipeline

    contract = build_prompt_contract(
        language,
        base_primary_id=gemini_asr.PROMPT_ID,
        base_primary_prompt=gemini_asr._PROMPT,
        base_quality_id=gemini_asr.QUALITY_PROMPT_ID,
        base_quality_prompt=gemini_asr._QUALITY_PROMPT,
    )
    gemini_asr.PROMPT_ID = contract.primary_id
    gemini_asr.PROMPT_SHA256 = contract.primary_sha256
    gemini_asr._PROMPT = contract.primary_prompt
    gemini_asr.QUALITY_PROMPT_ID = contract.quality_id
    gemini_asr.QUALITY_PROMPT_SHA256 = contract.quality_sha256
    gemini_asr._QUALITY_PROMPT = contract.quality_prompt

    pipeline.PROMPT_ID = contract.primary_id
    pipeline.PROMPT_SHA256 = contract.primary_sha256
    pipeline.QUALITY_PROMPT_ID = contract.quality_id
    if hasattr(pipeline, "QUALITY_PROMPT_SHA256"):
        pipeline.QUALITY_PROMPT_SHA256 = contract.quality_sha256

    # Older deployments called this set VIDEO_SUFFIXES.  Adding audio suffixes
    # here keeps the language runner compatible with both old and current code.
    media_suffixes: set[str] = getattr(
        pipeline,
        "MEDIA_SUFFIXES",
        getattr(pipeline, "VIDEO_SUFFIXES", set()),
    )
    media_suffixes.update({".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".mka"})
    if hasattr(pipeline, "MEDIA_SUFFIXES"):
        pipeline.MEDIA_SUFFIXES = media_suffixes
    else:
        pipeline.VIDEO_SUFFIXES = media_suffixes
    return contract


def duration_priority_key(
    duration_seconds: float,
    bands: Iterable[tuple[float, float | None]],
) -> int:
    duration = max(0.0, float(duration_seconds))
    materialized = list(bands)
    for index, (minimum, maximum) in enumerate(materialized):
        if duration >= minimum and (maximum is None or duration < maximum):
            return index
    return len(materialized)


_BAND_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*-\s*(\d+(?:\.\d+)?)?\s*$")


def parse_duration_priority_bands(value: str) -> tuple[tuple[float, float | None], ...]:
    bands: list[tuple[float, float | None]] = []
    if not str(value or "").strip():
        return ()
    for raw in str(value).split(","):
        match = _BAND_RE.match(raw)
        if not match:
            raise ValueError(f"invalid duration priority band: {raw!r}")
        minimum = float(match.group(1) or 0.0)
        maximum = float(match.group(2)) if match.group(2) else None
        if maximum is not None and maximum <= minimum:
            raise ValueError(f"invalid duration priority band: {raw!r}")
        bands.append((minimum, maximum))
    return tuple(bands)


__all__ = [
    "LANGUAGE_PROFILES",
    "LanguageProfile",
    "PromptContract",
    "activate_pipeline_language",
    "build_prompt_contract",
    "duration_priority_key",
    "get_language_profile",
    "normalized_title_key",
    "parse_duration_priority_bands",
    "script_code",
    "script_metrics",
]
