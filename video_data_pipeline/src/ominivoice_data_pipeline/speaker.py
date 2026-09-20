from __future__ import annotations

import math
from typing import Any

def _prefix_speaker_ids(candidates: list[dict[str, Any]], video_id: str) -> list[dict[str, Any]]:
    mapping: dict[str, str] = {}
    result: list[dict[str, Any]] = []
    for raw in candidates:
        item = dict(raw)
        original = str(item.get("speaker_id") or item.get("speaker") or "unknown").strip() or "unknown"
        if original == "unknown":
            stable = "unknown"
        else:
            mapping.setdefault(original, f"{video_id}_spk{len(mapping) + 1:02d}")
            stable = mapping[original]
        item["speaker_label"] = str(item.get("speaker_label") or original)
        item["speaker_id"] = stable
        item["speaker"] = stable
        item.setdefault("overlap_detected", False)
        result.append(item)
    return result


def label_candidates_with_turns(
    candidates: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    *,
    video_id: str,
    overlap_intervals: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Project optional acoustic diarization turns onto candidate segments."""
    labeled: list[dict[str, Any]] = []
    for raw in candidates:
        item = dict(raw)
        start = int(item.get("start_time", 0) or 0)
        end = int(item.get("end_time", 0) or 0)
        duration_ms = max(0, end - start)
        overlap_by_speaker: dict[str, int] = {}
        acoustic_overlap_ms = 0
        for interval in list(overlap_intervals or []):
            interval_start = int(interval.get("start_ms", interval.get("start_time", 0)) or 0)
            interval_end = int(interval.get("end_ms", interval.get("end_time", 0)) or 0)
            acoustic_overlap_ms += max(0, min(end, interval_end) - max(start, interval_start))
        acoustic_overlap_ratio = acoustic_overlap_ms / duration_ms if duration_ms > 0 else 0.0
        # Match the downstream hard gate: tiny collars are auditable but do not
        # prevent otherwise safe same-speaker merging.
        overlap_flag = acoustic_overlap_ms > max(20, round(duration_ms * 0.02))
        confidence_by_speaker: dict[str, list[float]] = {}
        confidence_source_by_speaker: dict[str, list[str]] = {}
        for turn in turns:
            turn_start = int(turn.get("start_ms", turn.get("start_time", 0)) or 0)
            turn_end = int(turn.get("end_ms", turn.get("end_time", 0)) or 0)
            overlap = max(0, min(end, turn_end) - max(start, turn_start))
            if overlap <= 0:
                continue
            state = str(turn.get("speaker_state") or turn.get("state") or "").strip().lower()
            raw_speakers = turn.get("speaker_ids")
            if isinstance(raw_speakers, list):
                speakers = [str(value).strip() for value in raw_speakers if str(value).strip()]
            else:
                speaker = str(turn.get("speaker_id") or turn.get("speaker") or "unknown").strip() or "unknown"
                speakers = [] if state in {"unknown", "unassigned"} or speaker == "unknown" else [speaker]
            raw_confidence = turn.get("confidence")
            confidence: float | None = None
            if not isinstance(raw_confidence, bool):
                try:
                    parsed_confidence = float(raw_confidence)
                    if math.isfinite(parsed_confidence):
                        confidence = max(0.0, min(1.0, parsed_confidence))
                except (TypeError, ValueError):
                    confidence = None
            confidence_source = str(
                turn.get("confidence_source")
                or turn.get("speaker_confidence_source")
                or ("turn_confidence" if confidence is not None else "unavailable")
            ).strip() or "unavailable"
            for speaker in speakers:
                overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0) + overlap
                if confidence is not None:
                    confidence_by_speaker.setdefault(speaker, []).append(confidence)
                    confidence_source_by_speaker.setdefault(speaker, []).append(confidence_source)
            overlap_flag = overlap_flag or state == "overlap" or bool(
                turn.get("overlap") or turn.get("overlap_detected")
            )
        if not overlap_by_speaker:
            item.update(
                {
                    "speaker_id": "unknown",
                    "speaker": "unknown",
                    "speaker_confidence": 0.0,
                    "speaker_confidence_source": "no_speaker_evidence",
                }
            )
        else:
            ordered = sorted(overlap_by_speaker.items(), key=lambda pair: (-pair[1], pair[0]))
            winner, winning_ms = ordered[0]
            second_ms = ordered[1][1] if len(ordered) > 1 else 0
            item["speaker_id"] = winner
            item["speaker"] = winner
            winner_confidences = confidence_by_speaker.get(winner) or []
            if winner_confidences:
                item["speaker_confidence"] = min(winner_confidences)
                item["speaker_confidence_source"] = "+".join(
                    dict.fromkeys(confidence_source_by_speaker.get(winner) or ["turn_confidence"])
                )
            else:
                item["speaker_confidence"] = None
                item["speaker_confidence_label"] = "unavailable"
                item["speaker_confidence_source"] = "unavailable_in_acoustic_turns"
            if second_ms > 0 and second_ms >= max(100, round(winning_ms * 0.15)):
                overlap_flag = True
        item["overlap_detected"] = overlap_flag
        item["acoustic_overlap_ms"] = acoustic_overlap_ms
        item["acoustic_overlap_ratio"] = round(acoustic_overlap_ratio, 6)
        item["speaker_assignment_source"] = "acoustic_turns"
        labeled.append(item)
    return _prefix_speaker_ids(labeled, video_id)
