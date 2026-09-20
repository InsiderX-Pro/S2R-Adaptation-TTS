"""Isolated pyannote-audio 4.x JSON-lines worker.

Run this file with the Python executable from the dedicated pyannote environment.
It loads one local pipeline snapshot, announces readiness, and then serves
diarization requests from stdin until a close request or EOF.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable


STARTUP_REQUEST_ID = "__startup__"
_PROTOCOL_STDOUT = sys.stdout


def _annotation_from_result(result: Any) -> Any:
    """Support pyannote returns that wrap the Annotation in a result object."""

    annotation = getattr(result, "speaker_diarization", None)
    return annotation if annotation is not None else result


def _raw_tracks(result: Any) -> list[dict[str, Any]]:
    annotation = _annotation_from_result(result)
    itertracks = getattr(annotation, "itertracks", None)
    if not callable(itertracks):
        raise TypeError("pyannote result does not expose itertracks(yield_label=True)")

    tracks: list[dict[str, Any]] = []
    for position, item in enumerate(itertracks(yield_label=True)):
        try:
            segment, track, label = item
            start = float(segment.start)
            end = float(segment.end)
        except (TypeError, ValueError, AttributeError) as exc:
            raise TypeError(f"invalid pyannote track at position {position}: {item!r}") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        speaker_id = str(label).strip()
        if not speaker_id:
            continue
        tracks.append(
            {
                "start": start,
                "end": end,
                "speaker_id": speaker_id,
                "track_id": str(track),
                "position": position,
            }
        )
    return sorted(
        tracks,
        key=lambda row: (row["start"], row["end"], row["speaker_id"], row["track_id"], row["position"]),
    )


def _overlap_flags(tracks: list[dict[str, Any]]) -> list[bool]:
    """Mark turns that overlap any simultaneously active different speaker."""

    flags = [False] * len(tracks)
    active: list[int] = []
    for index, current in enumerate(tracks):
        current_start = float(current["start"])
        active = [other for other in active if float(tracks[other]["end"]) > current_start]
        for other in active:
            previous = tracks[other]
            if previous["speaker_id"] == current["speaker_id"]:
                continue
            if min(float(previous["end"]), float(current["end"])) > max(
                float(previous["start"]), current_start
            ):
                flags[other] = True
                flags[index] = True
        active.append(index)
    return flags


def annotation_to_turns(result: Any) -> list[dict[str, Any]]:
    """Convert an Annotation or ``.speaker_diarization`` wrapper to JSON turns."""

    tracks = _raw_tracks(result)
    overlap = _overlap_flags(tracks)
    turns: list[dict[str, Any]] = []
    for index, track in enumerate(tracks):
        start_ms = int(round(float(track["start"]) * 1000.0))
        end_ms = int(round(float(track["end"]) * 1000.0))
        if end_ms <= start_ms:
            continue
        turns.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_time": start_ms,
                "end_time": end_ms,
                "speaker_id": str(track["speaker_id"]),
                "confidence_source": "unavailable",
                "overlap": bool(overlap[index]),
            }
        )
    return turns


def _derive_exclusive_turns(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a deterministic non-overlapping timeline for older pyannote results.

    Community-1 exposes an exclusive diarization directly.  Older pipelines and
    direct ``Annotation`` objects do not, so each atomic overlap is assigned to
    the speaker with the greatest total regular-track support.  A lexical label
    tie-break keeps the JSON stable across runs.
    """

    if not turns:
        return []
    support: dict[str, int] = {}
    boundaries: set[int] = set()
    for turn in turns:
        start_ms = int(turn["start_ms"])
        end_ms = int(turn["end_ms"])
        if end_ms <= start_ms:
            continue
        speaker_id = str(turn["speaker_id"])
        support[speaker_id] = support.get(speaker_id, 0) + end_ms - start_ms
        boundaries.update((start_ms, end_ms))

    exclusive: list[dict[str, Any]] = []
    ordered = sorted(boundaries)
    for start_ms, end_ms in zip(ordered, ordered[1:]):
        if end_ms <= start_ms:
            continue
        midpoint = (start_ms + end_ms) / 2.0
        active = {
            str(turn["speaker_id"])
            for turn in turns
            if int(turn["start_ms"]) <= midpoint < int(turn["end_ms"])
        }
        if not active:
            continue
        speaker_id = min(active, key=lambda label: (-support.get(label, 0), label))
        if (
            exclusive
            and exclusive[-1]["speaker_id"] == speaker_id
            and exclusive[-1]["end_ms"] == start_ms
        ):
            exclusive[-1]["end_ms"] = end_ms
            exclusive[-1]["end_time"] = end_ms
            continue
        exclusive.append(
            {
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_time": start_ms,
                "end_time": end_ms,
                "speaker_id": speaker_id,
                "confidence_source": "unavailable",
                "overlap": False,
            }
        )
    return exclusive


def diarization_to_timelines(result: Any) -> dict[str, Any]:
    """Return regular and exclusive JSON turns from a pyannote result.

    ``turns`` always represents regular diarization and therefore retains true
    overlaps.  ``exclusive_turns`` uses Community-1's native exclusive annotation
    when present, with a deterministic fallback for old/direct annotations.
    """

    turns = annotation_to_turns(result)
    exclusive_annotation = getattr(result, "exclusive_speaker_diarization", None)
    if exclusive_annotation is None:
        exclusive_turns = _derive_exclusive_turns(turns)
        source = "derived_from_regular"
    else:
        exclusive_turns = annotation_to_turns(exclusive_annotation)
        for turn in exclusive_turns:
            turn["overlap"] = False
        source = "pyannote_exclusive"
    return {
        "turns": turns,
        "exclusive_turns": exclusive_turns,
        "exclusive_source": source,
    }


def load_offline_pipeline(model_dir: str | Path, device: str) -> Any:
    """Load a local pyannote 4.x snapshot without allowing network fallback."""

    snapshot = Path(model_dir).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pyannote model snapshot does not exist: {snapshot}")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from pyannote.audio import Pipeline

    pipeline = Pipeline.from_pretrained(str(snapshot))
    if pipeline is None:
        raise RuntimeError(f"Pipeline.from_pretrained returned None for {snapshot}")
    move_to = getattr(pipeline, "to", None)
    if callable(move_to):
        try:
            import torch

            target_device: Any = torch.device(device)
        except ImportError:
            target_device = device
        move_to(target_device)
    return pipeline


def _speaker_count(turns: Iterable[dict[str, Any]]) -> int:
    return len({str(turn["speaker_id"]) for turn in turns})


def _diarize(pipeline: Any, audio_path: str, *, model_dir: str, device: str) -> dict[str, Any]:
    source = Path(audio_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"audio file does not exist: {source}")
    started = time.monotonic()
    with redirect_stdout(sys.stderr):
        result = pipeline(str(source))
    timelines = diarization_to_timelines(result)
    turns = timelines["turns"]
    exclusive_turns = timelines["exclusive_turns"]
    return {
        "status": "ok",
        "audio_path": str(source),
        "turns": turns,
        "exclusive_turns": exclusive_turns,
        "exclusive_source": timelines["exclusive_source"],
        "speaker_count": _speaker_count(turns),
        "overlap_turn_count": sum(bool(turn["overlap"]) for turn in turns),
        "model_dir": model_dir,
        "device": device,
        "elapsed_s": round(time.monotonic() - started, 3),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _emit(payload: dict[str, Any]) -> None:
    print(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        file=_PROTOCOL_STDOUT,
        flush=True,
    )


def main() -> int:
    args = build_parser().parse_args()
    model_dir = str(Path(args.model_dir).expanduser().resolve())
    try:
        with redirect_stdout(sys.stderr):
            pipeline = load_offline_pipeline(model_dir, args.device)
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        _emit(
            {
                "request_id": STARTUP_REQUEST_ID,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "model_dir": model_dir,
                "device": args.device,
            }
        )
        return 2

    _emit(
        {
            "request_id": STARTUP_REQUEST_ID,
            "status": "ready",
            "model_dir": model_dir,
            "device": args.device,
        }
    )
    for raw_line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            decoded = json.loads(raw_line)
            if not isinstance(decoded, dict):
                raise TypeError("request must be a JSON object")
            request = decoded
            request_id = str(request.get("request_id") or "")
            if not request_id:
                raise ValueError("request_id is required")
            operation = str(request.get("op") or "")
            if operation == "close":
                _emit({"request_id": request_id, "status": "closed"})
                break
            if operation != "diarize":
                raise ValueError(f"unsupported operation: {operation!r}")
            response = _diarize(
                pipeline,
                str(request["audio_path"]),
                model_dir=model_dir,
                device=args.device,
            )
            response["request_id"] = request_id
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {
                "request_id": str(request.get("request_id") or ""),
                "status": "error",
                "turns": [],
                "exclusive_turns": [],
                "error": f"{type(exc).__name__}: {exc}",
                "model_dir": model_dir,
                "device": args.device,
            }
        _emit(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "annotation_to_turns",
    "build_parser",
    "diarization_to_timelines",
    "load_offline_pipeline",
    "main",
]
