from __future__ import annotations

import json
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Iterable

from .io_utils import safe_id, sha256_file


def _require_binary(name: str) -> str:
    resolved = shutil.which(name)
    if not resolved:
        raise RuntimeError(f"{name} is required but was not found on PATH")
    return resolved


def _run(command: list[str], *, timeout_s: float = 3600.0) -> subprocess.CompletedProcess[str]:
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout_s)
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed ({process.returncode}): {' '.join(command[:4])}\n"
            f"stderr={str(process.stderr or '')[-4000:]}"
        )
    return process


def probe_media(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    process = _run(
        [
            _require_binary("ffprobe"),
            "-v",
            "error",
            "-show_entries",
            "format=filename,duration,size:stream=index,codec_type,codec_name,sample_rate,channels,channel_layout",
            "-of",
            "json",
            str(source),
        ],
        timeout_s=120.0,
    )
    payload = json.loads(process.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError(f"ffprobe returned invalid JSON for {source}")
    return payload


def _ffmpeg_audio(source: Path, destination: Path, extra: list[str]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            _require_binary("ffmpeg"),
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            *extra,
            str(destination),
        ]
    )
    if not destination.is_file() or destination.stat().st_size <= 44:
        raise RuntimeError(f"ffmpeg did not create a usable WAV: {destination}")
    return destination


def extract_audio_variants(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    analysis_sample_rate: int = 16000,
    dataset_sample_rate: int = 24000,
    keep_master: bool = True,
    resume: bool = False,
) -> dict[str, str]:
    video = Path(video_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    master = output / "master.wav"
    analysis = output / "analysis_16k_mono.wav"
    dataset = output / f"dataset_{int(dataset_sample_rate)}_mono.wav"

    if keep_master and not (resume and master.is_file()):
        _ffmpeg_audio(video, master, ["-c:a", "pcm_s16le"])
    if not (resume and analysis.is_file()):
        _ffmpeg_audio(
            video,
            analysis,
            ["-ar", str(int(analysis_sample_rate)), "-ac", "1", "-c:a", "pcm_s16le"],
        )
    if not (resume and dataset.is_file()):
        _ffmpeg_audio(
            video,
            dataset,
            ["-ar", str(int(dataset_sample_rate)), "-ac", "1", "-c:a", "pcm_s16le"],
        )
    return {
        "master_audio": str(master) if keep_master else "",
        "analysis_audio": str(analysis),
        "dataset_audio": str(dataset),
    }


def wav_info(path: str | Path) -> dict[str, int]:
    source = Path(path).expanduser().resolve()
    with wave.open(str(source), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return {
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "sample_rate": rate,
            "frame_count": frames,
            "duration_ms": round(frames * 1000 / rate),
        }


def export_wav_segments(
    source_wav: str | Path,
    segments: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    id_field: str = "segment_id",
    start_field: str = "start_ms",
    end_field: str = "end_ms",
    suffix_field: str | None = None,
) -> list[dict[str, Any]]:
    source = Path(source_wav).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        rate = reader.getframerate()
        total_frames = reader.getnframes()
        for position, raw in enumerate(segments, start=1):
            item = dict(raw)
            start_ms = max(0, int(item[start_field]))
            end_ms = max(start_ms, int(item[end_field]))
            start_frame = min(total_frames, round(start_ms * rate / 1000))
            end_frame = min(total_frames, round(end_ms * rate / 1000))
            reader.setpos(start_frame)
            frames = reader.readframes(max(0, end_frame - start_frame))
            item_id = safe_id(str(item.get(id_field) or f"segment_{position:06d}"))
            suffix = safe_id(str(item.get(suffix_field) or ""), fallback="") if suffix_field else ""
            filename = f"{item_id}_{suffix}.wav" if suffix else f"{item_id}.wav"
            destination = output / filename
            with wave.open(str(destination), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(frames)
            item["audio_path"] = str(destination)
            item["audio_sha256"] = sha256_file(destination)
            item["clip_start_ms"] = start_ms
            item["clip_end_ms"] = end_ms
            exported.append(item)
    return exported

