#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()[-4000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}\n{detail}")


def _verify_file(path: Path, expected_sha256: str, *, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    observed = _sha256(path)
    if observed != expected_sha256:
        raise RuntimeError(f"{label} hash mismatch: expected={expected_sha256}, observed={observed}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one cached BS-RoFormer model over a batch of short clips.")
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.expanduser().resolve().read_text(encoding="utf-8"))
    response_path = args.response.expanduser().resolve()

    package_version = str(request["audio_separator_version"])
    observed_version = importlib.metadata.version("audio-separator")
    if observed_version != package_version:
        raise RuntimeError(
            f"audio-separator version mismatch: expected={package_version}, observed={observed_version}"
        )
    from audio_separator.separator import Separator

    model_dir = Path(str(request["model_file_dir"])).expanduser().resolve()
    model_filename = str(request["model_filename"])
    model_path = model_dir / model_filename
    model_config_path = model_path.with_suffix(".yaml")
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(str(request["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = str(request.get("ffmpeg") or "ffmpeg")
    dataset_sample_rate = int(request["dataset_sample_rate"])
    minimum_padded_seconds = float(request["minimum_padded_seconds"])

    with tempfile.TemporaryDirectory(prefix="bs_roformer_", dir=output_dir) as temporary_raw:
        temporary = Path(temporary_raw)
        separator = Separator(
            log_level=30,
            model_file_dir=str(model_dir),
            output_dir=str(temporary),
            output_format="WAV",
            output_single_stem="Vocals",
            sample_rate=44_100,
            use_autocast=bool(request.get("use_autocast", True)),
        )
        separator.load_model(model_filename=model_filename)
        _verify_file(model_path, str(request["model_sha256"]), label="BS-RoFormer checkpoint")
        _verify_file(
            model_config_path,
            str(request["model_config_sha256"]),
            label="BS-RoFormer model config",
        )

        results: list[dict[str, Any]] = []
        for raw in list(request.get("items") or []):
            item = dict(raw)
            item_id = str(item["item_id"])
            source = Path(str(item["audio_path"])).expanduser().resolve()
            destination = Path(str(item["output_path"])).expanduser().resolve()
            duration_ms = int(item["duration_ms"])
            padded_seconds = max(minimum_padded_seconds, duration_ms / 1000.0 + 0.25)
            safe_stem = hashlib.sha256(item_id.encode("utf-8")).hexdigest()[:20]
            padded = temporary / f"{safe_stem}_padded.wav"
            raw_vocals = temporary / f"{safe_stem}_vocals.wav"
            try:
                _run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(source),
                        "-af",
                        "apad",
                        "-t",
                        f"{padded_seconds:.6f}",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-c:a",
                        "pcm_s16le",
                        str(padded),
                    ]
                )
                outputs = separator.separate(str(padded), {"Vocals": raw_vocals.stem})
                if raw_vocals.name not in {Path(str(value)).name for value in outputs} or not raw_vocals.is_file():
                    raise RuntimeError(f"BS-RoFormer did not emit the requested vocal stem: {outputs}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary_destination = destination.with_suffix(".tmp.wav")
                _run(
                    [
                        ffmpeg,
                        "-nostdin",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(raw_vocals),
                        "-t",
                        f"{duration_ms / 1000.0:.6f}",
                        "-ar",
                        str(dataset_sample_rate),
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        str(temporary_destination),
                    ]
                )
                os.replace(temporary_destination, destination)
                results.append(
                    {
                        "item_id": item_id,
                        "status": "ok",
                        "output_path": str(destination),
                        "output_sha256": _sha256(destination),
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "item_id": item_id,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            finally:
                for path in (padded, raw_vocals, destination.with_suffix(".tmp.wav")):
                    if path.is_file():
                        path.unlink()

    payload = {
        "schema_version": "ominivoice.bs-roformer-batch.v1",
        "audio_separator_version": observed_version,
        "model_filename": model_filename,
        "model_sha256": str(request["model_sha256"]),
        "items": results,
    }
    response_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(response_path, payload)
    print(json.dumps({"response": str(response_path), "item_count": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
