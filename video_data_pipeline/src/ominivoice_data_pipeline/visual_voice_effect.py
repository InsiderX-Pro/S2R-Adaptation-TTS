from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np


_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "fps": 2.0,
    "frame_width": 320,
    "frame_height": 180,
    "card_region": [0.0625, 0.238889, 0.28125, 0.761111],
    "navy_min_pixels": 3500,
    "cyan_min_pixels": 300,
    "cyan_max_pixels": 900,
    "merge_gap_ms": 1000,
    "min_interval_ms": 1500,
    "min_clip_overlap_ratio": 0.5,
    "min_clip_overlap_ms": 500,
}


def frame_has_identity_protection_card(frame: np.ndarray, config: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = {**_DEFAULT_CONFIG, **dict(config or {})}
    height, width = frame.shape[:2]
    left, top, right, bottom = [float(value) for value in selected["card_region"]]
    crop = frame[
        max(0, round(top * height)) : min(height, round(bottom * height)),
        max(0, round(left * width)) : min(width, round(right * width)),
    ]
    if crop.size == 0:
        raise ValueError("voice-effect card region is empty")
    red = crop[:, :, 0].astype(np.float32)
    green = crop[:, :, 1].astype(np.float32)
    blue = crop[:, :, 2].astype(np.float32)
    navy = (red < 45) & (green < 75) & (blue > 55) & (blue > red * 1.8) & (blue > green * 1.15)
    cyan = (red < 100) & (green > 80) & (blue > 110) & (blue > red * 1.4) & (green > red * 1.2)
    navy_pixels = int(navy.sum())
    cyan_pixels = int(cyan.sum())
    detected = (
        navy_pixels >= int(selected["navy_min_pixels"])
        and cyan_pixels >= int(selected["cyan_min_pixels"])
        and cyan_pixels <= int(selected["cyan_max_pixels"])
    )
    return {"detected": detected, "navy_pixels": navy_pixels, "cyan_pixels": cyan_pixels}


def detect_identity_protection_intervals(
    video_path: str | Path,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = {**_DEFAULT_CONFIG, **dict(config or {})}
    source = Path(video_path).expanduser().resolve()
    if not bool(selected["enabled"]):
        return {"schema_version": "ominivoice.visual-voice-effect.v1", "enabled": False, "intervals": []}
    fps = float(selected["fps"])
    width = int(selected["frame_width"])
    height = int(selected["frame_height"])
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("voice-effect visual fps and frame dimensions must be positive")
    process = subprocess.Popen(
        [
            "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"fps={fps},scale={width}:{height}", "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    frame_bytes = width * height * 3
    positive_ms: list[int] = []
    frame_index = 0
    while True:
        raw = process.stdout.read(frame_bytes)
        if not raw:
            break
        if len(raw) != frame_bytes:
            process.kill()
            raise RuntimeError("ffmpeg returned a truncated RGB frame")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        observation = frame_has_identity_protection_card(frame, selected)
        if observation["detected"]:
            positive_ms.append(round(frame_index * 1000.0 / fps))
        frame_index += 1
    stderr = (process.stderr.read() if process.stderr is not None else b"").decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"visual voice-effect ffmpeg failed ({return_code}): {stderr[-2000:]}")

    frame_ms = round(1000.0 / fps)
    merge_gap_ms = int(selected["merge_gap_ms"])
    intervals: list[list[int]] = []
    for timestamp_ms in positive_ms:
        if not intervals or timestamp_ms > intervals[-1][1] + merge_gap_ms:
            intervals.append([timestamp_ms, timestamp_ms + frame_ms])
        else:
            intervals[-1][1] = timestamp_ms + frame_ms
    min_interval_ms = int(selected["min_interval_ms"])
    rows = [
        {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": end_ms - start_ms,
            "reason": "broadcast_identity_protection_card",
            "voice_anonymization_suspected": True,
            "pitch_shift_suspected": True,
            "vocoder_effect_suspected": True,
        }
        for start_ms, end_ms in intervals
        if end_ms - start_ms >= min_interval_ms
    ]
    return {
        "schema_version": "ominivoice.visual-voice-effect.v1",
        "enabled": True,
        "source_video": str(source),
        "config": selected,
        "frame_count": frame_index,
        "positive_frame_count": len(positive_ms),
        "intervals": rows,
    }


def apply_identity_protection_review(
    items: list[dict[str, Any]],
    report: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selected = {**_DEFAULT_CONFIG, **dict(config or {})}
    intervals = [dict(row) for row in list(report.get("intervals") or []) if isinstance(row, dict)]
    review_all = bool(report.get("review_all"))
    review_all_reason = str(report.get("review_reason") or "visual_review_unavailable")
    output: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        start_ms = int(item.get("export_start_ms", item.get("start_ms", 0)) or 0)
        end_ms = int(item.get("export_end_ms", item.get("end_ms", 0)) or 0)
        duration_ms = max(1, end_ms - start_ms)
        matches: list[dict[str, Any]] = []
        if review_all:
            matches.append(
                {
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": duration_ms,
                    "reason": review_all_reason,
                    "overlap_ms": duration_ms,
                    "overlap_ratio": 1.0,
                }
            )
        for interval in intervals:
            overlap_ms = max(
                0,
                min(end_ms, int(interval["end_ms"])) - max(start_ms, int(interval["start_ms"])),
            )
            overlap_ratio = overlap_ms / duration_ms
            if overlap_ms >= min(int(selected["min_clip_overlap_ms"]), duration_ms) and overlap_ratio >= float(
                selected["min_clip_overlap_ratio"]
            ):
                matches.append({**interval, "overlap_ms": overlap_ms, "overlap_ratio": round(overlap_ratio, 6)})
        review_required = bool(matches)
        if review_required:
            item["voice_effect_review_required"] = True
            item["voice_effect_review_reasons"] = list(
                dict.fromkeys(
                    "broadcast_identity_protection_card_detected"
                    if str(match.get("reason") or "") == "broadcast_identity_protection_card"
                    else str(match.get("reason") or "visual_review_required")
                    for match in matches
                )
            )
            item["voice_effect_visual_matches"] = matches
            item["asr_eligible"] = False
            item["asr_ineligible_reasons"] = list(
                dict.fromkeys(list(item.get("asr_ineligible_reasons") or []) + ["voice_effect_review_required"])
            )
            eligibility = dict(item.get("eligibility") or {})
            eligibility.update(
                {
                    "asr_eligible": False,
                    "voice_effect_review_required": True,
                    "voice_effect_reasons": ["broadcast_identity_protection_card_detected"],
                }
            )
            item["eligibility"] = eligibility
        else:
            item["voice_effect_review_required"] = False
            item["voice_effect_review_reasons"] = []
            item["voice_effect_visual_matches"] = []
            eligibility = dict(item.get("eligibility") or {})
            eligibility.update({"voice_effect_review_required": False, "voice_effect_reasons": []})
            item["eligibility"] = eligibility
        output.append(item)
    return output


__all__ = [
    "apply_identity_protection_review",
    "detect_identity_protection_intervals",
    "frame_has_identity_protection_card",
]
