from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .io_utils import write_json, write_jsonl


def _merge_text(left: str, right: str) -> str:
    left = str(left or "").strip()
    right = str(right or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left[-1].isalnum() and right[0].isalnum():
        return f"{left} {right}"
    return left + right


def _speaker(item: dict[str, Any]) -> str:
    return str(item.get("speaker_id") or item.get("speaker") or "unknown").strip() or "unknown"


def _speaker_confidence(item: dict[str, Any]) -> float | None:
    raw = item.get("speaker_confidence")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _group_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_duration_ms: int,
    max_gap_ms: int,
) -> list[list[dict[str, Any]]]:
    ordered = sorted(
        (dict(item) for item in candidates),
        key=lambda item: (int(item.get("start_time", 0) or 0), int(item.get("end_time", 0) or 0)),
    )
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for item in ordered:
        if not current:
            current = [item]
            continue
        gap_ms = int(item.get("start_time", 0) or 0) - int(current[-1].get("end_time", 0) or 0)
        projected_ms = int(item.get("end_time", 0) or 0) - int(current[0].get("start_time", 0) or 0)
        can_merge = (
            _speaker(item) == _speaker(current[-1])
            and gap_ms <= max_gap_ms
            and projected_ms <= max_duration_ms
            and not bool(item.get("overlap_detected"))
            and not bool(current[-1].get("overlap_detected"))
        )
        if can_merge:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    if current:
        groups.append(current)
    return groups


def build_dataset_items(
    labeled_candidates: list[dict[str, Any]],
    *,
    source_video: str | Path,
    source_sha256: str,
    video_id: str,
    max_duration_ms: int = 15_000,
    max_gap_ms: int = 700,
    min_duration_ms: int = 1_000,
    asr_min_duration_ms: int = 500,
    min_speaker_confidence: float = 0.67,
) -> list[dict[str, Any]]:
    """Build final clip rows without crossing a speaker or overlap boundary.

    Eligibility dimensions are deliberately orthogonal:

    * ``speaker_pure`` describes only speaker identity/overlap confidence;
    * ``duration_valid`` describes the configured duration window;
    * ``audio_valid`` describes whether the candidate has a structurally valid
      audio span (and honors an explicit candidate ``audio_valid=False``);
    * ``short_answer`` marks clips below the dataset duration floor but at or
      above the lower ASR floor;
    * ``asr_eligible`` uses the ASR duration floor, so a clear short answer can
      still receive final ASR before a downstream dataset decision.

    ``speaker_accepted`` is retained as the backwards-compatible speaker gate
    field, but now mirrors ``speaker_pure`` instead of conflating speaker,
    duration, first-pass text, and audio-span quality.
    """
    groups = _group_candidates(
        labeled_candidates,
        max_duration_ms=max_duration_ms,
        max_gap_ms=max_gap_ms,
    )
    source = str(Path(source_video).expanduser().resolve())
    rows: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        start_ms = int(group[0].get("start_time", 0) or 0)
        end_ms = int(group[-1].get("end_time", 0) or 0)
        speaker_ids = sorted({_speaker(item) for item in group})
        confidences = [
            confidence
            for item in group
            if (confidence := _speaker_confidence(item)) is not None
        ]
        text = ""
        candidate_indices: list[int] = []
        for item in group:
            text = _merge_text(text, str(item.get("text") or ""))
            try:
                candidate_indices.append(int(item.get("candidate_index")))
            except (TypeError, ValueError):
                pass
        duration_ms = max(0, end_ms - start_ms)
        flags: list[str] = []
        speaker_flags: list[str] = []
        duration_flags: list[str] = []
        audio_flags: list[str] = []
        if not text:
            flags.append("empty_firstpass_text")
        if speaker_ids == ["unknown"] or not speaker_ids:
            speaker_flags.append("unknown_speaker")
        if len(speaker_ids) != 1:
            speaker_flags.append("multiple_speakers")
        if any(bool(item.get("overlap_detected")) for item in group):
            speaker_flags.append("overlap_detected")
        gate_reasons = sorted(
            {
                str(reason)
                for candidate in group
                for reason in list(dict(candidate.get("gate") or {}).get("reasons") or [])
                if str(reason)
            }
        )
        if gate_reasons:
            speaker_flags.append("speaker_gate_failed")
        if duration_ms < min_duration_ms:
            duration_flags.append("too_short")
        if duration_ms > max_duration_ms:
            duration_flags.append("too_long")
        confidence_available = bool(confidences)
        confidence = min(confidences) if confidences else None
        if confidence is not None and confidence < min_speaker_confidence:
            speaker_flags.append("low_speaker_confidence")
        explicit_audio_valid = all(bool(item.get("audio_valid", True)) for item in group)
        if start_ms < 0 or end_ms <= start_ms or not explicit_audio_valid:
            audio_flags.append("invalid_audio_span")
        flags.extend(speaker_flags)
        flags.extend(duration_flags)
        flags.extend(audio_flags)
        speaker_pure = not speaker_flags
        duration_valid = not duration_flags
        audio_valid = not audio_flags
        short_answer = asr_min_duration_ms <= duration_ms < min_duration_ms
        asr_duration_valid = asr_min_duration_ms <= duration_ms <= max_duration_ms
        asr_duration_reasons: list[str] = []
        if duration_ms < asr_min_duration_ms:
            asr_duration_reasons.append("below_asr_min_duration")
        if duration_ms > max_duration_ms:
            asr_duration_reasons.append("too_long")
        asr_eligible = speaker_pure and asr_duration_valid and audio_valid
        asr_ineligible_reasons = speaker_flags + asr_duration_reasons + audio_flags
        speaker_id = speaker_ids[0] if len(speaker_ids) == 1 else "unknown"
        rows.append(
            {
                "item_id": f"{video_id}_item_{index:06d}",
                "video_id": video_id,
                "source_video": source,
                "source_sha256": source_sha256,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": duration_ms,
                "speaker_id": speaker_id,
                "speaker_confidence": round(confidence, 4) if confidence is not None else None,
                "speaker_confidence_available": confidence_available,
                "speaker_confidence_observation_count": len(confidences),
                "speaker_ids_observed": speaker_ids,
                "candidate_indices": sorted(set(candidate_indices)),
                "speaker_gate_reasons": gate_reasons,
                "firstpass_text": text,
                "quality_flags": flags,
                "speaker_quality_flags": speaker_flags,
                "duration_quality_flags": duration_flags,
                "audio_quality_flags": audio_flags,
                "speaker_pure": speaker_pure,
                "duration_valid": duration_valid,
                "short_answer": short_answer,
                "asr_duration_valid": asr_duration_valid,
                "audio_valid": audio_valid,
                "asr_eligible": asr_eligible,
                "asr_eligibility_flags": asr_duration_reasons,
                "asr_ineligible_reasons": asr_ineligible_reasons,
                "eligibility": {
                    "speaker_pure": speaker_pure,
                    "duration_valid": duration_valid,
                    "short_answer": short_answer,
                    "asr_duration_valid": asr_duration_valid,
                    "audio_valid": audio_valid,
                    "asr_eligible": asr_eligible,
                    "speaker_reasons": speaker_flags,
                    "duration_reasons": duration_flags,
                    "asr_duration_reasons": asr_duration_reasons,
                    "audio_reasons": audio_flags,
                },
                "speaker_accepted": speaker_pure,
                "review_required": bool(flags),
                "review_status": "pending" if flags else "not_required",
                "accepted": False,
            }
        )
    return rows


def write_dataset_manifests(items: list[dict[str, Any]], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir).expanduser().resolve()
    json_path = write_json(output / "dataset_items.json", {"items": items})
    jsonl_path = write_jsonl(output / "manifest.jsonl", items)
    return json_path, jsonl_path
