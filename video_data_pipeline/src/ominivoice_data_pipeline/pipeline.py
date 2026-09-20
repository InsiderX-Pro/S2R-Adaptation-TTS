from __future__ import annotations

import hashlib
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .background_music_recovery import BackgroundMusicRecovery
from .dataset import build_dataset_items, write_dataset_manifests
from .gemini_asr import (
    PROMPT_ID,
    PROMPT_SHA256,
    QUALITY_PROMPT_ID,
    QUALITY_PROMPT_SHA256,
    GeminiConfig,
    GeminiTranscriber,
)
from .io_utils import (
    canonical_json_sha256,
    read_json,
    read_jsonl,
    safe_id,
    sha256_file,
    utc_now_iso,
    write_json,
    write_jsonl,
)
from .media import export_wav_segments, extract_audio_variants, probe_media, wav_info
from .music_detection import AudioSetMusicDetector, apply_background_music_quality
from .pyannote_diarization import PyannoteConfig, PyannoteDiarizer
from .quality import analyze_wav_quality, text_quality_report
from .segmentation import (
    energy_vad,
    find_quiet_boundary_ms,
    final_single_speaker_gate,
    fuse_timelines,
    prepare_speaker_timelines,
)
from .speaker import label_candidates_with_turns
from .visual_voice_effect import apply_identity_protection_review, detect_identity_protection_intervals


log = logging.getLogger(__name__)
MEDIA_SUFFIXES = {
    ".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi",
    ".wav", ".flac", ".mp3", ".m4a", ".ogg", ".opus",
}


@dataclass(frozen=True)
class PipelineConfig:
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    pyannote: PyannoteConfig | None = None
    analysis_sample_rate: int = 16_000
    dataset_sample_rate: int = 24_000
    keep_master: bool = True
    resume: bool = False
    vad: dict[str, Any] = field(default_factory=dict)
    items: dict[str, Any] = field(default_factory=dict)
    speaker_timeline: dict[str, Any] = field(default_factory=dict)
    speaker_gate: dict[str, Any] = field(default_factory=dict)
    audio_quality: dict[str, Any] = field(default_factory=dict)
    background_music: dict[str, Any] = field(default_factory=dict)
    background_music_recovery: dict[str, Any] = field(default_factory=dict)
    text_quality: dict[str, Any] = field(default_factory=dict)
    short_policy: dict[str, Any] = field(default_factory=dict)
    voice_effect_visual: dict[str, Any] = field(default_factory=dict)
    speaker_turns_dir: str | None = None
    asr_on_speaker_rejected: bool = False


def discover_videos(inputs: Iterable[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()
    for raw in inputs:
        source = Path(raw).expanduser().resolve()
        if source.is_file():
            candidates = [source]
        elif source.is_dir():
            candidates = sorted(path for path in source.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(source)
        for path in candidates:
            if path.suffix.lower() not in MEDIA_SUFFIXES or path in seen:
                continue
            seen.add(path)
            discovered.append(path)
    if not discovered:
        raise ValueError("no supported media files were found")
    return discovered


def _config_payload(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    # Resume controls I/O behavior, not the scientific/data contract.
    payload.pop("resume", None)
    # Preserve compatibility with runs created before the optional recovery
    # branch existed when it is explicitly disabled.
    recovery = dict(payload.get("background_music_recovery") or {})
    if not recovery or not bool(recovery.get("enabled", False)):
        payload.pop("background_music_recovery", None)
    return payload


def _validate_production_gemini(config: PipelineConfig) -> None:
    gemini = config.gemini
    expected = {
        "model": "gemini-2.5-flash",
        "backend": "formal_rest",
        "prompt_id": PROMPT_ID,
        "prompt_sha256": PROMPT_SHA256,
        "seeds": (101,),
        "max_attempts": 1,
        "temperature": 0.0,
        "thinking_budget": 0,
        "top_p": None,
        "top_k": 1,
        "max_output_tokens": 1024,
        "quality_enabled": True,
    }
    observed = {
        "model": gemini.model,
        "backend": gemini.backend,
        "prompt_id": gemini.prompt_id,
        "prompt_sha256": hashlib.sha256(gemini.prompt.encode("utf-8")).hexdigest(),
        "seeds": tuple(gemini.seeds),
        "max_attempts": gemini.max_attempts,
        "temperature": gemini.temperature,
        "thinking_budget": gemini.thinking_budget,
        "top_p": gemini.top_p,
        "top_k": gemini.top_k,
        "max_output_tokens": gemini.max_output_tokens,
        "quality_enabled": gemini.quality_enabled,
    }
    if observed != expected:
        raise ValueError(f"production Gemini configuration mismatch: expected={expected}, observed={observed}")


def _prepare_run(output_dir: str | Path, config: PipelineConfig) -> Path:
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not config.resume:
        raise FileExistsError(f"output directory is not empty; use --resume or a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    config_payload = _config_payload(config)
    config_hash = canonical_json_sha256(config_payload)
    config_path = output / "run_config.json"
    if config_path.is_file():
        previous = read_json(config_path)
        if str(previous.get("config_sha256") or "") != config_hash:
            raise RuntimeError("resume configuration does not match the existing run_config.json")
    else:
        write_json(
            config_path,
            {
                "schema_version": "ominivoice.run-config.v1",
                "created_at": utc_now_iso(),
                "config_sha256": config_hash,
                "config": config_payload,
            },
        )
    return output


def _sidecar(directory: str | None, video: Path, video_id: str) -> Path | None:
    if not directory:
        return None
    root = Path(directory).expanduser().resolve()
    for name in (f"{video_id}.json", f"{video.stem}.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no sidecar JSON for {video.name} in {root}")


def _normalize_speaker_timeline_payload(
    payload: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Normalize fresh pyannote and legacy/round-tripped sidecar shapes."""

    normalized = dict(payload)
    base_turns = normalized.get("turns", normalized.get("segments"))
    regular = normalized.get("regular_turns", normalized.get("regular_timeline"))
    exclusive = normalized.get(
        "exclusive_turns",
        normalized.get("assignment_turns", normalized.get("assignment_timeline")),
    )
    overlaps = normalized.get("overlap_intervals")
    if regular is None and isinstance(base_turns, list):
        regular = [dict(row) for row in base_turns if isinstance(row, dict)]
        # New atomic sidecars store assignment turns in `turns` and overlap
        # evidence separately. Reconstruct the regular evidence for old readers.
        if isinstance(overlaps, list):
            regular.extend(dict(row) for row in overlaps if isinstance(row, dict))
    if exclusive is None and isinstance(base_turns, list):
        exclusive = base_turns
    if not isinstance(regular, list):
        raise ValueError("speaker timeline payload needs regular turns/turns/segments")
    if not isinstance(exclusive, list):
        raise ValueError("speaker timeline payload needs exclusive or assignment turns")
    normalized["turns"] = [dict(row) for row in regular if isinstance(row, dict)]
    normalized["exclusive_turns"] = [dict(row) for row in exclusive if isinstance(row, dict)]
    normalized["config"] = dict(config)
    return normalized


def _vad_segments_for_export(video_id: str, vad_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, segment in enumerate(list(vad_payload.get("segments") or []), start=1):
        rows.append(
            {
                **dict(segment),
                "segment_id": f"{video_id}_vad_{index:06d}",
                "start_ms": round(float(segment.get("start", 0.0)) * 1000),
                "end_ms": round(float(segment.get("end", 0.0)) * 1000),
            }
        )
    return rows


def _fused_segments_for_export(
    video_id: str,
    vad_payload: dict[str, Any],
    speaker_payload: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Coarse ASR chunks use the exclusive assignment timeline. The regular
    # timeline is kept intact for the later purity/overlap gate.
    turns = speaker_payload.get(
        "assignment_timeline",
        speaker_payload.get("exclusive_turns", speaker_payload.get("turns", speaker_payload.get("segments"))),
    )
    if not isinstance(turns, list):
        raise ValueError("speaker timeline payload must contain a turns or segments list")
    fusion = fuse_timelines(
        {
            "vad_segments": _vad_segments_for_export(video_id, vad_payload),
            "speaker_segments": [dict(row) for row in turns if isinstance(row, dict)],
            "config": dict(config or {}),
        }
    )
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(list(fusion.get("segments") or []), start=1):
        segment = dict(raw)
        segment["segment_id"] = f"{video_id}_spk_atom_{index:06d}"
        segment["overlap_detected"] = str(segment.get("speaker_state") or "") == "overlap"
        rows.append(segment)
    return rows, fusion


def _speaker_gate(
    candidates: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    speaker_timeline: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    segment_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    stable_speakers: dict[str, str] = {}
    for index, candidate in enumerate(candidates):
        item = dict(candidate)
        segment_id = str(item.get("item_id") or item.get("candidate_index", index))
        row_id = segment_id if item.get("item_id") else f"candidate-{segment_id}"
        start_ms = int(item.get("start_ms", item.get("start_time", 0)) or 0)
        end_ms = int(item.get("end_ms", item.get("end_time", 0)) or 0)
        stable_speaker = str(item.get("speaker_id") or item.get("speaker") or "unknown")
        stable_speakers[row_id] = stable_speaker
        segment_rows.append(
            {
                **item,
                "id": row_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
        if speaker_timeline is None:
            state = "overlap" if bool(item.get("overlap_detected")) else "single"
            if stable_speaker == "unknown":
                state = "unknown"
            turn_rows.append(
                {
                    "id": f"turn-{segment_id}",
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "speaker_id": stable_speaker,
                    "speaker_state": state,
                }
            )
    gate_result = final_single_speaker_gate(
        {
            "segments": segment_rows,
            "speaker_segments": list(speaker_timeline) if speaker_timeline is not None else turn_rows,
            "config": dict(config),
        }
    )
    for row in list(gate_result.get("segments") or []):
        row_id = str(row.get("id") or "")
        row["acoustic_gate_speaker_id"] = row.get("speaker_id")
        row["speaker_id"] = stable_speakers.get(row_id, "unknown")
        row["speaker"] = row["speaker_id"]
    return gate_result


def _cap_candidate_durations(
    candidate_result: dict[str, Any],
    *,
    max_duration_ms: int,
    audio_path: str | Path | None = None,
    split_search_radius_ms: int = 1_500,
    min_piece_ms: int = 1_000,
) -> dict[str, Any]:
    """Split unaligned overlong candidates at nearby low-energy boundaries.

    If an overlong candidate is split, its previous coarse text cannot be
    aligned safely to the new sub-spans. Final Gemini ASR supplies each chunk's
    actual text.
    """

    if max_duration_ms <= 0:
        raise ValueError("max_duration_ms must be positive")
    if split_search_radius_ms < 0 or min_piece_ms <= 0 or min_piece_ms > max_duration_ms:
        raise ValueError("duration-cap split parameters must be positive")
    output = {key: value for key, value in candidate_result.items() if key != "utterances"}
    capped: list[dict[str, Any]] = []
    split_count = 0
    for raw in list(candidate_result.get("utterances") or []):
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        start = int(item.get("start_time", 0) or 0)
        end = int(item.get("end_time", 0) or 0)
        duration = max(0, end - start)
        part_count = max(1, math.ceil(duration / max_duration_ms))
        if part_count == 1:
            item["candidate_index"] = len(capped)
            capped.append(item)
            continue
        split_count += 1
        source_text = str(item.get("text") or "")
        source_candidate_index = item.get("candidate_index")
        boundaries = [start]
        boundary_diagnostics: list[dict[str, Any]] = []
        for boundary_index in range(1, part_count):
            remaining_parts = part_count - boundary_index
            hard_lower = max(boundaries[-1] + min_piece_ms, end - remaining_parts * max_duration_ms)
            hard_upper = min(boundaries[-1] + max_duration_ms, end - remaining_parts * min_piece_ms)
            target = start + round(duration * boundary_index / part_count)
            target = min(hard_upper, max(hard_lower, target))
            search_lower = max(hard_lower, target - split_search_radius_ms)
            search_upper = min(hard_upper, target + split_search_radius_ms)
            diagnostic: dict[str, Any] = {
                "target_ms": target,
                "search_start_ms": search_lower,
                "search_end_ms": search_upper,
                "method": "balanced_time_fallback",
                "rms_dbfs": None,
            }
            split = target
            if audio_path is not None and search_upper > search_lower:
                try:
                    quiet = find_quiet_boundary_ms(
                        audio_path,
                        target_ms=target,
                        lower_ms=search_lower,
                        upper_ms=search_upper,
                    )
                    split = int(quiet["split_ms"])
                    diagnostic.update(quiet)
                except (OSError, ValueError):
                    log.warning("quiet-boundary search failed; using balanced fallback", exc_info=True)
            split = min(hard_upper, max(hard_lower, split))
            diagnostic["split_ms"] = split
            boundaries.append(split)
            boundary_diagnostics.append(diagnostic)
        boundaries.append(end)
        for part_index in range(part_count):
            part_start = boundaries[part_index]
            part_end = boundaries[part_index + 1]
            part = dict(item)
            part.update(
                {
                    "candidate_index": len(capped),
                    "source_candidate_index": source_candidate_index,
                    "start_time": part_start,
                    "end_time": part_end,
                    "text": "",
                    "words": [],
                    "duration_cap_split": True,
                    "duration_cap_part_index": part_index,
                    "duration_cap_part_count": part_count,
                    "duration_cap_split_method": (
                        "sustained_pause_rms"
                        if any(row.get("method") == "sustained_pause_rms" for row in boundary_diagnostics)
                        else "balanced_time_fallback"
                    ),
                    "duration_cap_boundary_before": (
                        boundary_diagnostics[part_index - 1] if part_index > 0 else None
                    ),
                    "duration_cap_boundary_after": (
                        boundary_diagnostics[part_index] if part_index < len(boundary_diagnostics) else None
                    ),
                    "unaligned_source_text": source_text,
                    "text_alignment": "unavailable_after_time_cap",
                }
            )
            capped.append(part)
    output["utterances"] = capped
    output["text"] = " ".join(str(item.get("text") or "").strip() for item in capped).strip()
    output["duration_cap"] = {
        "max_duration_ms": max_duration_ms,
        "split_search_radius_ms": split_search_radius_ms,
        "min_piece_ms": min_piece_ms,
        "split_strategy": "sustained_pause_near_balanced_target",
        "source_candidates_split": split_count,
        "output_candidate_count": len(capped),
    }
    return output


def _apply_gate_eligibility(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refresh orthogonal eligibility fields after gating merged item spans."""

    refreshed: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        gate_reasons = [str(reason) for reason in list(dict(item.get("gate") or {}).get("reasons") or []) if str(reason)]
        speaker_flags = [
            str(reason)
            for reason in list(item.get("speaker_quality_flags") or [])
            if str(reason) and str(reason) != "speaker_gate_failed"
        ]
        if gate_reasons:
            speaker_flags.append("speaker_gate_failed")
        if any(
            reason == "multiple_speakers" or "other_speaker" in reason or "overlap" in reason
            for reason in gate_reasons
        ):
            speaker_flags.append("broadcast_crosstalk_detected")
        speaker_flags = list(dict.fromkeys(speaker_flags))
        duration_flags = [str(reason) for reason in list(item.get("duration_quality_flags") or []) if str(reason)]
        audio_flags = [str(reason) for reason in list(item.get("audio_quality_flags") or []) if str(reason)]
        non_dimension_flags = [
            str(reason)
            for reason in list(item.get("quality_flags") or [])
            if str(reason) not in set(speaker_flags + duration_flags + audio_flags + ["speaker_gate_failed"])
        ]
        speaker_pure = not speaker_flags
        asr_duration_valid = bool(item.get("asr_duration_valid"))
        audio_valid = bool(item.get("audio_valid"))
        asr_eligible = speaker_pure and asr_duration_valid and audio_valid
        asr_ineligible = list(speaker_flags)
        asr_ineligible.extend(str(reason) for reason in list(item.get("asr_eligibility_flags") or []) if str(reason))
        asr_ineligible.extend(audio_flags)
        eligibility = dict(item.get("eligibility") or {})
        eligibility.update(
            {
                "speaker_pure": speaker_pure,
                "asr_eligible": asr_eligible,
                "speaker_reasons": speaker_flags,
            }
        )
        item.update(
            {
                "speaker_gate_reasons": gate_reasons,
                "speaker_quality_flags": speaker_flags,
                "quality_flags": list(dict.fromkeys(non_dimension_flags + speaker_flags + duration_flags + audio_flags)),
                "speaker_pure": speaker_pure,
                "speaker_accepted": speaker_pure,
                "asr_eligible": asr_eligible,
                "asr_ineligible_reasons": list(dict.fromkeys(asr_ineligible)),
                "eligibility": eligibility,
            }
        )
        refreshed.append(item)
    return refreshed


def _apply_local_audio_quality(
    items: list[dict[str, Any]],
    *,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    checked: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        report = analyze_wav_quality(item["audio_path"], config)
        audio_flags = list(
            dict.fromkeys(
                [str(reason) for reason in list(item.get("audio_quality_flags") or []) if str(reason)]
                + [str(reason) for reason in list(report.get("reasons") or []) if str(reason)]
            )
        )
        audio_valid = bool(item.get("audio_valid", True)) and not audio_flags
        speaker_pure = bool(item.get("speaker_pure"))
        asr_duration_valid = bool(item.get("asr_duration_valid"))
        asr_eligible = speaker_pure and asr_duration_valid and audio_valid
        eligibility = dict(item.get("eligibility") or {})
        eligibility.update({"audio_valid": audio_valid, "asr_eligible": asr_eligible, "audio_reasons": audio_flags})
        existing_quality = [
            str(reason)
            for reason in list(item.get("quality_flags") or [])
            if str(reason) and str(reason) not in set(item.get("audio_quality_flags") or [])
        ]
        ineligible = [
            str(reason)
            for reason in list(item.get("asr_ineligible_reasons") or [])
            if str(reason) and str(reason) not in set(item.get("audio_quality_flags") or [])
        ]
        item.update(
            {
                "local_audio_quality": report,
                "audio_quality_flags": audio_flags,
                "quality_flags": list(dict.fromkeys(existing_quality + audio_flags)),
                "audio_valid": audio_valid,
                "asr_eligible": asr_eligible,
                "asr_ineligible_reasons": list(dict.fromkeys(ineligible + audio_flags)),
                "eligibility": eligibility,
            }
        )
        checked.append(item)
    return checked


def _batch_gemini_final(
    items: list[dict[str, Any]],
    transcriber: GeminiTranscriber,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, item in enumerate(items, start=1):
        result = transcriber.transcribe_clip(item["audio_path"])
        rows.append({"item_id": item["item_id"], "audio_sha256": item["audio_sha256"], **result})
        if index == 1 or index % 25 == 0 or index == len(items):
            elapsed = max(0.001, time.monotonic() - started)
            rate = index / elapsed
            remaining_minutes = (len(items) - index) / rate / 60.0 if rate > 0 else 0.0
            log.info(
                "Gemini ASR progress %s/%s (%.2f clips/min, ETA %.1f min)",
                index,
                len(items),
                rate * 60.0,
                remaining_minutes,
            )
    return rows


def _single_successful_primary(row: dict[str, Any]) -> dict[str, Any] | None:
    runs = [dict(run) for run in list(row.get("runs") or []) if isinstance(run, dict)]
    successful = [run for run in runs if run.get("status") == "ok"]
    return successful[0] if len(runs) == 1 and len(successful) == 1 else None


def _batch_gemini_quality(
    items: list[dict[str, Any]],
    gemini_rows: list[dict[str, Any]],
    transcriber: GeminiTranscriber,
) -> list[dict[str, Any]]:
    """Run an independent observer without changing the frozen P3 ASR request."""

    primary_by_id = {str(row.get("item_id") or ""): row for row in gemini_rows}
    quality_inputs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in items:
        primary = _single_successful_primary(primary_by_id.get(str(item["item_id"]), {}))
        if primary is not None:
            quality_inputs.append((item, primary))

    rows: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, (item, primary) in enumerate(quality_inputs, start=1):
        transcript = str(primary.get("transcript") or "")
        result = transcriber.assess_clip(item["audio_path"], transcript)
        rows.append(
            {
                "item_id": item["item_id"],
                "audio_sha256": item["audio_sha256"],
                "primary_transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                **result,
            }
        )
        if index == 1 or index % 25 == 0 or index == len(quality_inputs):
            elapsed = max(0.001, time.monotonic() - started)
            rate = index / elapsed
            remaining_minutes = (len(quality_inputs) - index) / rate / 60.0 if rate > 0 else 0.0
            log.info(
                "Gemini quality progress %s/%s (%.2f clips/min, ETA %.1f min)",
                index,
                len(quality_inputs),
                rate * 60.0,
                remaining_minutes,
            )
    return rows


def _finalize_gemini_only(
    items: list[dict[str, Any]],
    gemini_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
    *,
    text_quality_config: dict[str, Any] | None = None,
    short_policy_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Finalize frozen P3 text using a separately persisted quality observation."""

    short_policy = {
        "drop_below_ms": 500,
        "clear_short_max_ms": 1_000,
        **dict(short_policy_config or {}),
    }
    drop_below_ms = int(short_policy["drop_below_ms"])
    clear_short_max_ms = int(short_policy["clear_short_max_ms"])
    if drop_below_ms < 0 or clear_short_max_ms <= drop_below_ms:
        raise ValueError("short policy requires 0 <= drop_below_ms < clear_short_max_ms")
    by_id = {str(row.get("item_id") or ""): row for row in gemini_rows}
    quality_by_id = {str(row.get("item_id") or ""): row for row in quality_rows}
    finalized: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        item_id = str(item["item_id"])
        intrinsic_eligible = bool(
            item.get("asr_eligible")
            if "asr_eligible" in item
            else item.get("speaker_accepted", True)
        )
        blocking = list(item.get("asr_ineligible_reasons") or []) if not intrinsic_eligible else []
        payload = by_id.get(item_id)
        quality_payload = quality_by_id.get(item_id)
        runs = [dict(row) for row in list((payload or {}).get("runs") or []) if isinstance(row, dict)]
        successful = [row for row in runs if row.get("status") == "ok"]
        selected = successful[0] if len(successful) == 1 else {}
        transcript_report = text_quality_report(
            str(selected.get("transcript") or ""),
            text_quality_config,
        )
        transcript = str(transcript_report["normalized_text"])
        asr_failure_reasons: list[str] = []
        quality_failure_reasons: list[str] = []
        quality_reasons: list[str] = []
        quality_warning_reasons: list[str] = []
        boundary_repair_reasons: list[str] = []
        if payload is None:
            asr_failure_reasons.append("gemini_asr_not_run")
        elif len(runs) != 1 or len(successful) != 1:
            asr_failure_reasons.append("gemini_asr_failed")
        else:
            quality_reasons.extend(str(reason) for reason in list(transcript_report["reasons"]) if str(reason))
            guard = dict(selected.get("guard") or {})
            guard_reasons = [str(reason) for reason in list(guard.get("reasons") or []) if str(reason)]
            if "pathological_repetition" in guard_reasons:
                quality_reasons.append("repetitive_text")
            quality_reasons.extend(reason for reason in guard_reasons if reason != "pathological_repetition")
            assessment = dict((quality_payload or {}).get("quality_assessment") or {})
            if quality_payload is None:
                quality_failure_reasons.append("gemini_quality_not_run")
            elif quality_payload.get("status") != "ok" or not assessment:
                quality_failure_reasons.append("gemini_quality_failed")
            else:
                if assessment.get("speech_complete") is not True:
                    boundary_repair_reasons.append("incomplete_or_cut_off_speech")
                for field, reason in (
                    ("breath_or_noise_only", "breath_or_noise_only"),
                    ("background_music", "background_music_detected"),
                    ("sound_effects", "sound_effects_detected"),
                    ("broadcast_crosstalk", "broadcast_crosstalk_detected"),
                    ("voice_disguise_effect", "voice_disguise_effect_detected"),
                    ("low_snr", "low_snr_detected"),
                    ("clipping", "clipping_detected"),
                ):
                    if assessment.get(field) is True:
                        quality_reasons.append(reason)
                text_metrics = dict(transcript_report.get("metrics") or {})
                latin_count = int(text_metrics.get("latin_letter_count", 0) or 0)
                latin_ratio = float(text_metrics.get("latin_ratio", 0.0) or 0.0)
                min_unnatural_latin_ratio = float(
                    dict(text_quality_config or {}).get("min_unnatural_latin_ratio", 0.35)
                )
                if latin_count:
                    if assessment.get("english_natural_code_switch") is not True:
                        if latin_ratio >= min_unnatural_latin_ratio:
                            quality_reasons.append("unnatural_english")
                        else:
                            quality_warning_reasons.append("english_code_switch_review")
                    if assessment.get("english_excessive") is True and "excessive_english" not in quality_reasons:
                        quality_warning_reasons.append("gemini_english_excessive_below_text_ratio_limit")
                # Audio/text disagreement is objective enough to fail closed and
                # applies even when the transcript contains no Latin characters.
                if assessment.get("english_audio_text_consistent") is not True:
                    quality_reasons.append("english_audio_text_mismatch")
                duration_ms = int(item.get("duration_ms", 0) or 0)
                if duration_ms < drop_below_ms:
                    quality_reasons.append("below_short_policy_minimum")
                elif duration_ms < clear_short_max_ms and assessment.get("natural_short_response") is not True:
                    quality_reasons.append("unclear_or_unnatural_short_response")
        quality_reasons = list(dict.fromkeys(quality_reasons))
        quality_warning_reasons = list(dict.fromkeys(quality_warning_reasons))
        boundary_repair_reasons = list(dict.fromkeys(boundary_repair_reasons))
        quality_failure_reasons = list(dict.fromkeys(quality_failure_reasons))
        reasons = list(
            dict.fromkeys(
                blocking
                + asr_failure_reasons
                + quality_failure_reasons
                + boundary_repair_reasons
                + quality_reasons
            )
        )
        excluded = bool(blocking or quality_reasons)
        retry_required = not excluded and bool(asr_failure_reasons or quality_failure_reasons)
        boundary_repair_required = not excluded and not retry_required and bool(boundary_repair_reasons)
        accepted = not excluded and not retry_required and not boundary_repair_required
        item.update(
            {
                "gemini_text": transcript,
                "asr_model": str((payload or {}).get("model") or selected.get("model") or ""),
                "asr_prompt_id": str((payload or {}).get("prompt_id") or selected.get("prompt_id") or ""),
                "qa_required": bool(quality_warning_reasons or boundary_repair_reasons),
                "qa_reasons": reasons,
                "qa_route": (
                    "auto_accept_with_warning"
                    if accepted and quality_warning_reasons
                    else "auto_accept"
                    if accepted
                    else "boundary_repair"
                    if boundary_repair_required
                    else "excluded_from_dataset"
                    if excluded
                    else "asr_retry"
                ),
                "review_required": False,
                "review_status": (
                    "warning_only"
                    if accepted and quality_warning_reasons
                    else "not_required"
                    if accepted
                    else "boundary_repair_pending"
                    if boundary_repair_required
                    else "excluded"
                    if excluded
                    else "retry_pending"
                ),
                "retry_required": retry_required,
                "boundary_repair_required": boundary_repair_required,
                "boundary_repair_reasons": boundary_repair_reasons,
                "excluded": excluded,
                "discarded": False,
                "calibration_required": False,
                "calibration_candidate": False,
                "accepted": accepted,
                "short_answer_accepted": accepted and bool(item.get("short_answer")),
                "quality_passed": not quality_reasons and not boundary_repair_reasons,
                "quality_rejection_reasons": quality_reasons,
                "quality_warning_reasons": quality_warning_reasons,
                "text_quality": transcript_report,
                "quality_prompt_id": str((quality_payload or {}).get("prompt_id") or ""),
                "gemini_quality_assessment": dict((quality_payload or {}).get("quality_assessment") or {}),
                "final_text": transcript if accepted else "",
                "text_source": "gemini_flash" if accepted else "",
            }
        )
        finalized.append(item)
    return finalized


def _build_training_manifest(
    accepted: list[dict[str, Any]],
    *,
    short_policy_config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    policy = {
        "training_min_ratio": 0.02,
        "training_target_ratio": 0.03,
        "training_max_ratio": 0.05,
        **dict(short_policy_config or {}),
    }
    minimum = float(policy["training_min_ratio"])
    target = float(policy["training_target_ratio"])
    maximum = float(policy["training_max_ratio"])
    if not 0.0 <= minimum <= target <= maximum < 1.0:
        raise ValueError("short training ratios must satisfy 0 <= min <= target <= max < 1")

    standard = [dict(item) for item in accepted if not item.get("short_answer_accepted")]
    short = [dict(item) for item in accepted if item.get("short_answer_accepted")]
    selected_short_count = 0
    minimum_count = 0
    if standard and short and maximum > 0.0:
        maximum_count = math.floor(maximum * len(standard) / (1.0 - maximum))
        minimum_count = math.ceil(minimum * len(standard) / (1.0 - minimum)) if minimum > 0.0 else 0
        target_count = round(target * len(standard) / (1.0 - target))
        if maximum_count >= minimum_count:
            selected_short_count = min(len(short), maximum_count, max(minimum_count, target_count))
    ranked_short = sorted(
        short,
        key=lambda item: (
            hashlib.sha256(str(item.get("item_id") or "").encode("utf-8")).hexdigest(),
            str(item.get("item_id") or ""),
        ),
    )
    selected_ids = {str(item.get("item_id") or "") for item in ranked_short[:selected_short_count]}
    training: list[dict[str, Any]] = []
    for raw in accepted:
        is_short = bool(raw.get("short_answer_accepted"))
        selected = not is_short or str(raw.get("item_id") or "") in selected_ids
        if not selected:
            continue
        item = dict(raw)
        item["training_sampling_group"] = "short_500_999ms" if is_short else "standard_ge_1000ms"
        item["selected_for_training"] = True
        training.append(item)
    actual_ratio = selected_short_count / len(training) if training else 0.0
    return training, {
        "schema_version": "ominivoice.short-sampling.v1",
        "policy": {
            "minimum_ratio": minimum,
            "target_ratio": target,
            "maximum_ratio": maximum,
            "selection": "deterministic_item_id_hash",
        },
        "available_standard_count": len(standard),
        "available_short_count": len(short),
        "selected_standard_count": len(standard),
        "selected_short_count": selected_short_count,
        "training_count": len(training),
        "actual_short_ratio": round(actual_ratio, 6),
        "below_minimum_due_to_supply": bool(short and actual_ratio < minimum and len(short) < minimum_count),
    }


def _apply_speaker_safe_padding(
    items: list[dict[str, Any]],
    *,
    speaker_timeline: list[dict[str, Any]] | None,
    audio_duration_ms: int,
    padding_ms: int | None = None,
    left_padding_ms: int | None = None,
    right_padding_ms: int | None = None,
) -> list[dict[str, Any]]:
    """Add export context without crossing evidence for another/unknown speaker."""

    if padding_ms is not None:
        left_padding_ms = padding_ms if left_padding_ms is None else left_padding_ms
        right_padding_ms = padding_ms if right_padding_ms is None else right_padding_ms
    left_padding_ms = 0 if left_padding_ms is None else int(left_padding_ms)
    right_padding_ms = 0 if right_padding_ms is None else int(right_padding_ms)
    if left_padding_ms < 0 or right_padding_ms < 0:
        raise ValueError("audio padding must be non-negative")
    turns = [dict(row) for row in list(speaker_timeline or []) if isinstance(row, dict)]
    output: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        start = int(item.get("start_ms", item.get("start_time", 0)) or 0)
        end = int(item.get("end_ms", item.get("end_time", 0)) or 0)
        target_speaker = str(item.get("acoustic_gate_speaker_id") or "")
        export_start = start
        export_end = end
        policy = "disabled"
        if (left_padding_ms or right_padding_ms) and turns and target_speaker and target_speaker != "unknown":
            requested_start = max(0, start - left_padding_ms)
            requested_end = min(audio_duration_ms, end + right_padding_ms)
            export_start = requested_start
            export_end = requested_end
            policy = "speaker_safe_timeline"
            for turn in turns:
                turn_start = int(turn.get("start_ms", turn.get("start_time", 0)) or 0)
                turn_end = int(turn.get("end_ms", turn.get("end_time", 0)) or 0)
                speaker = str(turn.get("speaker_id", turn.get("speaker", "unknown")) or "unknown")
                state = str(turn.get("speaker_state", turn.get("state", "single")) or "single")
                conflict = state != "single" or speaker in {"", "unknown", "unassigned"} or speaker != target_speaker
                if not conflict:
                    continue
                if turn_start < start and turn_end > requested_start:
                    export_start = max(export_start, min(start, turn_end))
                if turn_end > end and turn_start < requested_end:
                    export_end = min(export_end, max(end, turn_start))
        item.update(
            {
                "export_start_ms": export_start,
                "export_end_ms": export_end,
                "clip_padding_left_ms": max(0, start - export_start),
                "clip_padding_right_ms": max(0, export_end - end),
                "padding_policy": policy,
            }
        )
        output.append(item)
    return output


def _select_asr_attempts(
    items: list[dict[str, Any]],
    *,
    include_speaker_rejected: bool,
) -> list[dict[str, Any]]:
    """Optionally relax only speaker purity, never duration/audio validity."""

    selected: list[dict[str, Any]] = []
    for item in items:
        if not bool(item.get("asr_duration_valid")) or not bool(item.get("audio_valid")):
            continue
        if bool(item.get("voice_effect_review_required")):
            continue
        if bool(item.get("asr_eligible")):
            selected.append(item)
            continue
        if not include_speaker_rejected:
            continue
        speaker_reasons = {
            str(reason)
            for reason in list(item.get("speaker_quality_flags") or [])
            if str(reason)
        }
        speaker_reasons.update({"speaker_gate_failed", "broadcast_crosstalk_detected"})
        blocking_reasons = {
            str(reason)
            for reason in list(item.get("asr_ineligible_reasons") or [])
            if str(reason)
        }
        if not blocking_reasons.difference(speaker_reasons):
            selected.append(item)
    return selected


def _resume_asr_rows(path: Path, items: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    reusable = _reusable_asr_rows(path, items)
    if len(reusable) != len(items):
        return None
    return [reusable[str(item["item_id"])] for item in items]


def _reusable_asr_rows(path: Path, items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    expected = {str(item["item_id"]): str(item["audio_sha256"]) for item in items}
    reusable: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = dict(raw)
        item_id = str(row.get("item_id") or "")
        if (
            item_id in expected
            and str(row.get("audio_sha256") or "") == expected[item_id]
            and str(row.get("prompt_id") or "") == PROMPT_ID
            and str(row.get("prompt_sha256") or "") == PROMPT_SHA256
            and _single_successful_primary(row) is not None
        ):
            reusable[item_id] = row
    return reusable


def _resume_quality_rows(
    path: Path,
    items: list[dict[str, Any]],
    gemini_rows: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    reusable = _reusable_quality_rows(path, items, gemini_rows)
    primary_by_id = {str(row.get("item_id") or ""): row for row in gemini_rows}
    expected_ids = [
        str(item["item_id"])
        for item in items
        if _single_successful_primary(primary_by_id.get(str(item["item_id"]), {})) is not None
    ]
    if len(reusable) != len(expected_ids):
        return None
    return [reusable[item_id] for item_id in expected_ids]


def _reusable_quality_rows(
    path: Path,
    items: list[dict[str, Any]],
    gemini_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    primary_by_id = {str(row.get("item_id") or ""): row for row in gemini_rows}
    expected: dict[str, tuple[str, str, str, str]] = {}
    for item in items:
        item_id = str(item["item_id"])
        primary = _single_successful_primary(primary_by_id.get(item_id, {}))
        if primary is None:
            continue
        transcript_sha256 = hashlib.sha256(str(primary.get("transcript") or "").encode("utf-8")).hexdigest()
        expected[item_id] = (
            str(item["audio_sha256"]),
            transcript_sha256,
            QUALITY_PROMPT_ID,
            QUALITY_PROMPT_SHA256,
        )
    reusable: dict[str, dict[str, Any]] = {}
    for raw in read_jsonl(path):
        row = dict(raw)
        item_id = str(row.get("item_id") or "")
        observed = (
            str(row.get("audio_sha256") or ""),
            str(row.get("primary_transcript_sha256") or ""),
            str(row.get("prompt_id") or ""),
            str(row.get("prompt_sha256") or ""),
        )
        if (
            item_id in expected
            and observed == expected[item_id]
            and row.get("status") == "ok"
            and bool(dict(row.get("quality_assessment") or {}))
        ):
            reusable[item_id] = row
    return reusable


def _publish_final_clips(
    accepted: list[dict[str, Any]],
    output_dir: str | Path,
) -> list[dict[str, Any]]:
    """Publish only accepted WAVs and point final rows at the clean directory.

    A hard link keeps publication cheap on the normal same-filesystem layout;
    copy2 is the portable fallback. Stale files are removed so a resumed run
    cannot leave a previously accepted clip in the final delivery directory.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_names: set[str] = set()
    published: list[dict[str, Any]] = []
    for raw in accepted:
        item = dict(raw)
        candidate = Path(str(item.get("audio_path") or "")).expanduser().resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"accepted candidate audio is missing: {candidate}")
        destination = output / candidate.name
        expected_sha256 = str(item.get("audio_sha256") or sha256_file(candidate))
        if destination.is_file() and sha256_file(destination) != expected_sha256:
            destination.unlink()
        if not destination.exists():
            try:
                os.link(candidate, destination)
                publication_method = "hardlink"
            except OSError:
                shutil.copy2(candidate, destination)
                publication_method = "copy"
        else:
            publication_method = "existing"
        if sha256_file(destination) != expected_sha256:
            raise RuntimeError(f"published final clip hash mismatch: {destination}")
        expected_names.add(destination.name)
        item.update(
            {
                "candidate_audio_path": str(candidate),
                "audio_path": str(destination),
                "audio_publication_method": publication_method,
            }
        )
        published.append(item)
    for stale in output.glob("*.wav"):
        if stale.name not in expected_names:
            stale.unlink()
    return published


def _run_one_video(
    video: Path,
    run_root: Path,
    config: PipelineConfig,
    gemini: GeminiTranscriber,
    diarizer: PyannoteDiarizer | None,
    music_detector: AudioSetMusicDetector,
    music_recovery: BackgroundMusicRecovery,
) -> dict[str, Any]:
    source_sha256 = sha256_file(video)
    video_id = f"{safe_id(video.stem)}-{source_sha256[:8]}"
    work = run_root / "videos" / video_id
    work.mkdir(parents=True, exist_ok=True)
    log.info("[%s] probing and extracting audio", video_id)
    probe = probe_media(video)
    write_json(work / "source.json", {"video_id": video_id, "source_video": str(video), "sha256": source_sha256, "probe": probe})
    audio = extract_audio_variants(
        video,
        work / "audio",
        analysis_sample_rate=config.analysis_sample_rate,
        dataset_sample_rate=config.dataset_sample_rate,
        keep_master=config.keep_master,
        resume=config.resume,
    )

    vad_path = work / "vad.json"
    if config.resume and vad_path.is_file():
        vad_payload = read_json(vad_path)
    else:
        log.info("[%s] running adaptive energy VAD", video_id)
        vad_payload = energy_vad({"audio_path": audio["analysis_audio"], "config": dict(config.vad)})
        write_json(vad_path, vad_payload)
    speaker_turns = _sidecar(config.speaker_turns_dir, video, video_id)
    speaker_payload: dict[str, Any] | None = None
    prepared_speaker: dict[str, Any] | None = None
    regular_speaker_timeline: list[dict[str, Any]] | None = None
    if speaker_turns is not None:
        speaker_payload = read_json(speaker_turns)
    elif diarizer is not None:
        pyannote_path = work / "speaker" / "pyannote_turns.json"
        if config.resume and pyannote_path.is_file():
            speaker_payload = read_json(pyannote_path)
        else:
            log.info("[%s] running offline pyannote diarization", video_id)
            speaker_payload = diarizer.diarize(audio["analysis_audio"])
            write_json(pyannote_path, speaker_payload)
        if speaker_payload.get("status") != "ok":
            raise RuntimeError(f"pyannote diarization failed: {speaker_payload.get('error')}")

    if speaker_payload is not None:
        timeline_payload = _normalize_speaker_timeline_payload(
            speaker_payload,
            config=dict(config.speaker_timeline),
        )
        prepared_speaker = prepare_speaker_timelines(timeline_payload)
        if not bool(prepared_speaker.get("assignment_valid")):
            raise RuntimeError(
                "speaker sidecar has no valid exclusive assignment timeline; "
                "provide pyannote exclusive_turns instead of overlapping regular turns"
            )
        if not list(prepared_speaker.get("assignment_timeline") or []):
            raise RuntimeError("speaker assignment timeline is empty")
        write_json(work / "speaker" / "prepared_speaker_timelines.json", prepared_speaker)
        regular_speaker_timeline = [
            dict(row) for row in list(prepared_speaker.get("regular_timeline") or []) if isinstance(row, dict)
        ]
        coarse, fusion = _fused_segments_for_export(
            video_id,
            vad_payload,
            prepared_speaker,
            config=dict(config.speaker_timeline),
        )
        write_json(work / "speaker_timeline_fusion.json", fusion)
        assignment_turns_path = work / "speaker" / "atomic_speaker_turns.json"
        write_json(
            assignment_turns_path,
            {
                "schema_version": "ominivoice.atomic-speaker-turns.v1",
                "source": str(speaker_turns) if speaker_turns is not None else "pyannote",
                "turns": list(prepared_speaker.get("assignment_timeline") or []),
                "regular_timeline": list(prepared_speaker.get("regular_timeline") or []),
                "overlap_intervals": list(prepared_speaker.get("overlap_intervals") or []),
                "assignment_valid": bool(prepared_speaker.get("assignment_valid")),
            },
        )
        speaker_turns = assignment_turns_path
    else:
        raise RuntimeError("Gemini-only production pipeline requires pyannote or --speaker-turns-dir")
    if not coarse:
        summary = {
            "video_id": video_id,
            "source_video": str(video),
            "status": "passed_no_speech",
            "item_count": 0,
        }
        write_json(work / "video_summary.json", summary)
        return summary
    candidate_result = {
        "schema_version": "ominivoice.vad-pyannote-candidates.v1",
        "duration": max((int(row.get("end_ms", 0) or 0) for row in coarse), default=0),
        "text": "",
        "utterances": [
            {
                **dict(row),
                "candidate_index": index,
                "start_time": int(row.get("start_ms", 0) or 0),
                "end_time": int(row.get("end_ms", 0) or 0),
                "text": "",
                "words": [],
                "segmentation_source": "vad_pyannote_fusion",
            }
            for index, row in enumerate(coarse)
        ],
    }
    candidate_result = _cap_candidate_durations(
        candidate_result,
        max_duration_ms=int(dict(config.items).get("max_duration_ms", 12_000)),
        audio_path=audio["analysis_audio"],
        split_search_radius_ms=int(dict(config.items).get("split_search_radius_ms", 1_500)),
        min_piece_ms=int(dict(config.items).get("split_min_piece_ms", 1_000)),
    )
    write_json(work / "speaker_candidate_utterances.json", candidate_result)
    candidate_sha256 = canonical_json_sha256(
        {
            "candidate_result": candidate_result,
            "speaker_turns_sha256": sha256_file(speaker_turns) if speaker_turns is not None else None,
        }
    )
    speaker_cache_path = work / "speaker" / "labeled_candidates_cache.json"
    speaker_cache = read_json(speaker_cache_path) if config.resume and speaker_cache_path.is_file() else {}
    if not isinstance(speaker_cache, dict):
        speaker_cache = {}
    cached_rows = speaker_cache.get("labeled")
    if speaker_cache.get("candidate_sha256") == candidate_sha256 and isinstance(cached_rows, list):
        labeled = [dict(row) for row in cached_rows if isinstance(row, dict)]
        speaker_metadata = dict(speaker_cache.get("metadata") or {})
    else:
        log.info("[%s] projecting pyannote speakers onto VAD candidates", video_id)
        assignment_timeline = [
            dict(row)
            for row in list((prepared_speaker or {}).get("assignment_timeline") or [])
            if isinstance(row, dict)
        ]
        overlap_intervals = [
            dict(row)
            for row in list((prepared_speaker or {}).get("overlap_intervals") or [])
            if isinstance(row, dict)
        ]
        labeled = label_candidates_with_turns(
            [dict(row) for row in list(candidate_result.get("utterances") or [])],
            assignment_timeline,
            video_id=video_id,
            overlap_intervals=overlap_intervals,
        )
        speaker_metadata = {
            "backend": "pyannote_acoustic_turns",
            "turn_count": len(assignment_timeline),
            "overlap_interval_count": len(overlap_intervals),
        }
        write_json(
            speaker_cache_path,
            {
                "schema_version": "ominivoice.speaker-cache.v1",
                "candidate_sha256": candidate_sha256,
                "labeled": labeled,
                "metadata": speaker_metadata,
            },
        )
    write_json(work / "labeled_candidates.json", {"candidates": labeled, "speaker": speaker_metadata})

    item_config = dict(config.items)
    items = build_dataset_items(
        labeled,
        source_video=video,
        source_sha256=source_sha256,
        video_id=video_id,
        max_duration_ms=int(item_config.get("max_duration_ms", 15_000)),
        max_gap_ms=int(item_config.get("max_gap_ms", 700)),
        min_duration_ms=int(item_config.get("min_duration_ms", 1_000)),
        asr_min_duration_ms=int(item_config.get("asr_min_duration_ms", 300)),
        min_speaker_confidence=float(item_config.get("min_speaker_confidence", 0.67)),
    )
    gate_result = _speaker_gate(
        items,
        config=dict(config.speaker_gate),
        speaker_timeline=regular_speaker_timeline,
    )
    items = _apply_gate_eligibility(
        [dict(row) for row in list(gate_result.get("segments") or [])]
    )
    gate_result["segments"] = items
    gate_result["accepted"] = [item for item in items if dict(item.get("gate") or {}).get("passed")]
    gate_result["rejected"] = [item for item in items if not dict(item.get("gate") or {}).get("passed")]
    write_json(work / "single_speaker_gate.json", gate_result)
    items = _apply_speaker_safe_padding(
        items,
        speaker_timeline=regular_speaker_timeline,
        left_padding_ms=int(item_config.get("audio_padding_left_ms", item_config.get("audio_padding_ms", 200))),
        right_padding_ms=int(item_config.get("audio_padding_right_ms", item_config.get("audio_padding_ms", 350))),
        audio_duration_ms=int(wav_info(audio["dataset_audio"])["duration_ms"]),
    )
    items = export_wav_segments(
        audio["dataset_audio"],
        items,
        work / "candidate_clips",
        id_field="item_id",
        suffix_field="speaker_id",
        start_field="export_start_ms",
        end_field="export_end_ms",
    )
    items = _apply_local_audio_quality(items, config=dict(config.audio_quality))
    music_path = work / "background_music.jsonl"
    if music_recovery.process_all:
        log.info("[%s] skipping AST probability detection; sending all clips to BS-RoFormer", video_id)
        music_rows = [
            {
                "item_id": str(item["item_id"]),
                "audio_sha256": str(item.get("audio_sha256") or ""),
                "schema_version": "ominivoice.background-music.v1",
                "enabled": False,
                "detected": False,
                "reasons": [],
                "analysis_skipped": True,
                "analysis_skip_reason": "direct_all_separation",
            }
            for item in items
        ]
        write_jsonl(music_path, music_rows)
    else:
        # Materialize the iterator: it is inspected for cache validity and then
        # reused below. Keeping the generator here would silently exhaust it.
        resumed_music = (
            [dict(row) for row in read_jsonl(music_path)]
            if config.resume and music_path.is_file()
            else []
        )
        expected_music = [(str(item["item_id"]), str(item.get("audio_sha256") or "")) for item in items]
        observed_music = [
            (str(row.get("item_id") or ""), str(row.get("audio_sha256") or "")) for row in resumed_music
        ]
        music_contract = {"enabled": bool(music_detector.config["enabled"])}
        if music_detector.enabled:
            music_contract.update(
                {
                    "model": str(music_detector.config["model"]),
                    "model_revision": str(music_detector.config["model_revision"]),
                    "label": str(music_detector.config["label"]),
                    "threshold": float(music_detector.config["threshold"]),
                    "sample_rate": int(music_detector.config["sample_rate"]),
                    "window_seconds": float(music_detector.config["window_seconds"]),
                    "hop_seconds": float(music_detector.config["hop_seconds"]),
                }
            )
        resumed_music_contract_valid = all(
            all(row.get(key) == value for key, value in music_contract.items()) for row in resumed_music
        )
        if observed_music != expected_music or not resumed_music_contract_valid:
            log.info("[%s] running local AudioSet AST background-music detection", video_id)
            music_rows = music_detector.analyze_items(items)
            write_jsonl(music_path, music_rows)
        else:
            music_rows = resumed_music
    recovery_path = work / "background_music_recovery.jsonl"
    cached_recovery = (
        [dict(row) for row in read_jsonl(recovery_path)]
        if config.resume and recovery_path.is_file()
        else []
    )
    detected_music_count = sum(bool(row.get("detected")) for row in music_rows)
    if music_recovery.process_all:
        log.info("[%s] running BS-RoFormer on all %s clips", video_id, len(items))
    elif detected_music_count:
        log.info(
            "[%s] attempting local BS-RoFormer recovery on %s music-positive clips",
            video_id,
            detected_music_count,
        )
    items, effective_music_rows, recovery_rows = music_recovery.recover_items(
        items,
        music_rows,
        output_dir=work / "recovered_clips",
        music_detector=music_detector,
        audio_quality_config=dict(config.audio_quality),
        cached_reports=cached_recovery,
        request_path=work / "bs_roformer_request.json",
        response_path=work / "bs_roformer_response.json",
    )
    write_jsonl(recovery_path, recovery_rows)
    items = apply_background_music_quality(items, effective_music_rows)
    voice_effect_path = work / "voice_effect_visual.json"
    if config.resume and voice_effect_path.is_file():
        voice_effect_report = read_json(voice_effect_path)
        if voice_effect_report.get("skip_reason") == "source_has_no_video_stream" and voice_effect_report.get(
            "review_all"
        ):
            voice_effect_report["review_all"] = False
            voice_effect_report["review_reason"] = "audio_quality_voice_disguise_guard"
            write_json(voice_effect_path, voice_effect_report)
    elif not any(
        isinstance(stream, dict) and stream.get("codec_type") == "video"
        for stream in list(probe.get("streams") or [])
    ):
        log.info("[%s] skipping visual voice-effect scan: source has no video stream", video_id)
        voice_effect_report = {
            "schema_version": "ominivoice.visual-voice-effect.v1",
            "enabled": bool(config.voice_effect_visual.get("enabled", True)),
            "source_video": str(video),
            "skipped": True,
            "skip_reason": "source_has_no_video_stream",
            "review_all": False,
            "review_reason": "audio_quality_voice_disguise_guard",
            "intervals": [],
        }
        write_json(voice_effect_path, voice_effect_report)
    else:
        log.info("[%s] scanning identity-protected interview visual intervals", video_id)
        voice_effect_report = detect_identity_protection_intervals(video, dict(config.voice_effect_visual))
        write_json(voice_effect_path, voice_effect_report)
    items = apply_identity_protection_review(items, voice_effect_report, dict(config.voice_effect_visual))
    write_jsonl(
        work / "audio_quality.jsonl",
        [
            {
                "item_id": item["item_id"],
                **dict(item.get("local_audio_quality") or {}),
            }
            for item in items
        ],
    )
    write_dataset_manifests(items, work)
    intrinsic_eligible = [item for item in items if item.get("asr_eligible")]
    eligible = _select_asr_attempts(
        items,
        include_speaker_rejected=config.asr_on_speaker_rejected,
    )

    gemini_path = work / "gemini_results.jsonl"
    resumed_gemini = _resume_asr_rows(gemini_path, eligible) if config.resume else None
    if resumed_gemini is None:
        reusable_gemini = _reusable_asr_rows(gemini_path, eligible) if config.resume else {}
        pending_gemini = [item for item in eligible if str(item["item_id"]) not in reusable_gemini]
        log.info(
            "[%s] running frozen Gemini Flash ASR on %s clips (%s reused)",
            video_id,
            len(pending_gemini),
            len(reusable_gemini),
        )
        fresh_gemini = _batch_gemini_final(pending_gemini, gemini)
        gemini_by_id = {**reusable_gemini, **{str(row["item_id"]): row for row in fresh_gemini}}
        gemini_rows = [gemini_by_id[str(item["item_id"])] for item in eligible]
        write_jsonl(gemini_path, gemini_rows)
    else:
        gemini_rows = resumed_gemini

    quality_path = work / "gemini_quality_results.jsonl"
    resumed_quality = (
        _resume_quality_rows(quality_path, eligible, gemini_rows)
        if config.resume and config.gemini.quality_enabled
        else None
    )
    if not config.gemini.quality_enabled:
        quality_rows: list[dict[str, Any]] = []
    elif resumed_quality is None:
        reusable_quality = _reusable_quality_rows(quality_path, eligible, gemini_rows) if config.resume else {}
        pending_quality = [item for item in eligible if str(item["item_id"]) not in reusable_quality]
        log.info(
            "[%s] running independent Gemini quality observer on %s clips (%s reused)",
            video_id,
            len(pending_quality),
            len(reusable_quality),
        )
        fresh_quality = _batch_gemini_quality(pending_quality, gemini_rows, gemini)
        quality_by_id = {**reusable_quality, **{str(row["item_id"]): row for row in fresh_quality}}
        primary_by_id = {str(row.get("item_id") or ""): row for row in gemini_rows}
        quality_rows = [
            quality_by_id[str(item["item_id"])]
            for item in eligible
            if _single_successful_primary(primary_by_id.get(str(item["item_id"]), {})) is not None
            and str(item["item_id"]) in quality_by_id
        ]
        write_jsonl(quality_path, quality_rows)
    else:
        quality_rows = resumed_quality

    finalized = _finalize_gemini_only(
        items,
        gemini_rows,
        quality_rows,
        text_quality_config=dict(config.text_quality),
        short_policy_config=dict(config.short_policy),
    )
    accepted_candidates = [item for item in finalized if item.get("accepted")]
    accepted_rows = _publish_final_clips(accepted_candidates, work / "final_clips")
    accepted_by_id = {str(item["item_id"]): item for item in accepted_rows}
    finalized = [accepted_by_id.get(str(item["item_id"]), item) for item in finalized]
    write_json(work / "dataset_items.json", {"items": finalized})
    write_jsonl(work / "manifest.jsonl", finalized)
    standard_rows = [item for item in accepted_rows if not item.get("short_answer_accepted")]
    short_answer_rows = [item for item in accepted_rows if item.get("short_answer_accepted")]
    retry_rows = [item for item in finalized if item.get("retry_required")]
    boundary_repair_rows = [item for item in finalized if item.get("boundary_repair_required")]
    voice_effect_review_rows = [item for item in finalized if item.get("voice_effect_review_required")]
    excluded_rows = [item for item in finalized if item.get("excluded")]
    write_jsonl(work / "final_manifest.jsonl", accepted_rows)
    write_jsonl(work / "standard_manifest.jsonl", standard_rows)
    write_jsonl(work / "short_answer_manifest.jsonl", short_answer_rows)
    write_jsonl(work / "asr_retry_queue.jsonl", retry_rows)
    write_jsonl(work / "boundary_repair_queue.jsonl", boundary_repair_rows)
    write_jsonl(work / "voice_effect_review_queue.jsonl", voice_effect_review_rows)
    write_jsonl(work / "excluded_manifest.jsonl", excluded_rows)
    summary = {
        "video_id": video_id,
        "source_video": str(video),
        "status": "passed",
        "asr_prompt_id": PROMPT_ID,
        "quality_prompt_id": QUALITY_PROMPT_ID,
        "vad_segment_count": len(coarse),
        "candidate_count": len(labeled),
        "item_count": len(finalized),
        "asr_eligible_count": len(intrinsic_eligible),
        "asr_attempted_count": len(eligible),
        "accepted_count": len(accepted_rows),
        "standard_accepted_count": len(standard_rows),
        "short_answer_accepted_count": len(short_answer_rows),
        "retry_count": len(retry_rows),
        "boundary_repair_count": len(boundary_repair_rows),
        "voice_effect_review_count": len(voice_effect_review_rows),
        "background_music_detected_count": sum(
            bool(dict(item.get("local_background_music") or {}).get("initial_detected"))
            or bool(dict(item.get("local_background_music") or {}).get("detected"))
            for item in finalized
        ),
        "background_music_recovered_count": sum(
            dict(item.get("background_music_recovery") or {}).get("status") == "recovered"
            for item in finalized
        ),
        "background_music_recovery_rejected_count": sum(
            dict(item.get("background_music_recovery") or {}).get("status") in {"rejected", "error"}
            for item in finalized
        ),
        "excluded_count": len(excluded_rows),
        "speaker_gate": gate_result.get("summary"),
        "output_dir": str(work),
    }
    write_json(work / "video_summary.json", summary)
    return summary


def run_pipeline(
    inputs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    config: PipelineConfig,
) -> dict[str, Any]:
    _validate_production_gemini(config)
    output_candidate = Path(output_dir).expanduser().resolve()
    videos = [
        video
        for video in discover_videos(inputs)
        if video != output_candidate and output_candidate not in video.parents
    ]
    if not videos:
        raise ValueError("no source videos remain after excluding the output directory")
    output = _prepare_run(output_dir, config)
    gemini = GeminiTranscriber(config.gemini)
    music_detector = AudioSetMusicDetector(dict(config.background_music))
    music_recovery = BackgroundMusicRecovery(
        dict(config.background_music_recovery),
        dataset_sample_rate=config.dataset_sample_rate,
    )
    diarizer = PyannoteDiarizer(config.pyannote) if config.pyannote is not None else None
    try:
        summaries: list[dict[str, Any]] = []
        failures: list[dict[str, str]] = []
        inventory: list[dict[str, Any]] = []
        for video in videos:
            source_hash = sha256_file(video)
            inventory.append(
                {
                    "path": str(video),
                    "name": video.name,
                    "size": video.stat().st_size,
                    "sha256": source_hash,
                }
            )
            try:
                summaries.append(
                    _run_one_video(
                        video,
                        output,
                        config,
                        gemini,
                        diarizer,
                        music_detector,
                        music_recovery,
                    )
                )
            except Exception as exc:
                log.exception("pipeline failed for %s", video)
                failures.append({"video": str(video), "error": f"{type(exc).__name__}: {exc}"})
        aggregate_final: list[dict[str, Any]] = []
        aggregate_standard: list[dict[str, Any]] = []
        aggregate_short_answers: list[dict[str, Any]] = []
        aggregate_retries: list[dict[str, Any]] = []
        aggregate_boundary_repairs: list[dict[str, Any]] = []
        aggregate_voice_effect_reviews: list[dict[str, Any]] = []
        aggregate_excluded: list[dict[str, Any]] = []
        for video_summary in summaries:
            video_output = str(video_summary.get("output_dir") or "")
            if not video_output:
                continue
            video_root = Path(video_output)
            if (video_root / "final_manifest.jsonl").is_file():
                aggregate_final.extend(dict(row) for row in read_jsonl(video_root / "final_manifest.jsonl"))
            for name, destination in (
                ("standard_manifest.jsonl", aggregate_standard),
                ("short_answer_manifest.jsonl", aggregate_short_answers),
                ("asr_retry_queue.jsonl", aggregate_retries),
                ("boundary_repair_queue.jsonl", aggregate_boundary_repairs),
                ("voice_effect_review_queue.jsonl", aggregate_voice_effect_reviews),
                ("excluded_manifest.jsonl", aggregate_excluded),
            ):
                if (video_root / name).is_file():
                    destination.extend(dict(row) for row in read_jsonl(video_root / name))
        write_jsonl(output / "inventory.jsonl", inventory)
        write_jsonl(output / "final_manifest.jsonl", aggregate_final)
        write_jsonl(output / "standard_manifest.jsonl", aggregate_standard)
        write_jsonl(output / "short_answer_manifest.jsonl", aggregate_short_answers)
        write_jsonl(output / "asr_retry_queue.jsonl", aggregate_retries)
        write_jsonl(output / "boundary_repair_queue.jsonl", aggregate_boundary_repairs)
        write_jsonl(output / "voice_effect_review_queue.jsonl", aggregate_voice_effect_reviews)
        write_jsonl(output / "excluded_manifest.jsonl", aggregate_excluded)
        training_rows, short_sampling = _build_training_manifest(
            aggregate_final,
            short_policy_config=dict(config.short_policy),
        )
        write_jsonl(output / "training_manifest.jsonl", training_rows)
        write_json(output / "short_sampling_summary.json", short_sampling)
        status = "passed" if not failures else "partial_failure"
        summary = {
            "schema_version": "ominivoice.run-summary.v1",
            "status": status,
            "completed_at": utc_now_iso(),
            "output_dir": str(output),
            "video_count": len(videos),
            "completed_video_count": len(summaries),
            "failed_video_count": len(failures),
            "item_count": sum(int(row.get("item_count", 0) or 0) for row in summaries),
            "asr_eligible_count": sum(int(row.get("asr_eligible_count", 0) or 0) for row in summaries),
            "asr_attempted_count": sum(int(row.get("asr_attempted_count", 0) or 0) for row in summaries),
            "accepted_count": sum(int(row.get("accepted_count", 0) or 0) for row in summaries),
            "standard_accepted_count": sum(
                int(row.get("standard_accepted_count", 0) or 0) for row in summaries
            ),
            "short_answer_accepted_count": sum(
                int(row.get("short_answer_accepted_count", 0) or 0) for row in summaries
            ),
            "retry_count": sum(int(row.get("retry_count", 0) or 0) for row in summaries),
            "boundary_repair_count": sum(
                int(row.get("boundary_repair_count", 0) or 0) for row in summaries
            ),
            "voice_effect_review_count": sum(
                int(row.get("voice_effect_review_count", 0) or 0) for row in summaries
            ),
            "background_music_detected_count": sum(
                int(row.get("background_music_detected_count", 0) or 0) for row in summaries
            ),
            "background_music_recovered_count": sum(
                int(row.get("background_music_recovered_count", 0) or 0) for row in summaries
            ),
            "background_music_recovery_rejected_count": sum(
                int(row.get("background_music_recovery_rejected_count", 0) or 0) for row in summaries
            ),
            "excluded_count": sum(int(row.get("excluded_count", 0) or 0) for row in summaries),
            "training_count": len(training_rows),
            "training_short_count": int(short_sampling["selected_short_count"]),
            "training_short_ratio": float(short_sampling["actual_short_ratio"]),
            "videos": summaries,
            "failures": failures,
        }
        write_json(output / "run_summary.json", summary)
        marker = "COMPLETE.json" if status == "passed" else "PARTIAL_FAILURE.json"
        stale_marker = output / ("PARTIAL_FAILURE.json" if status == "passed" else "COMPLETE.json")
        if stale_marker.is_file():
            stale_marker.unlink()
        write_json(
            output / marker,
            {"status": status, "completed_at": summary["completed_at"], "summary": "run_summary.json"},
        )
        if failures:
            raise RuntimeError(f"pipeline completed with {len(failures)} failed video(s); see run_summary.json")
        return summary
    finally:
        if diarizer is not None:
            diarizer.close()
