from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_json_config
from .gemini_asr import PROMPT_ID, GeminiConfig
from .pipeline import PipelineConfig, run_pipeline
from .pyannote_diarization import (
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_PYANNOTE_PYTHON,
    PyannoteConfig,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")


def _speaker_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pyannote-python",
        default=os.environ.get("PYANNOTE_PYTHON", DEFAULT_PYANNOTE_PYTHON),
        help="Python executable from the isolated pyannote environment",
    )
    parser.add_argument(
        "--pyannote-model",
        default=os.environ.get("PYANNOTE_MODEL_DIR", DEFAULT_PYANNOTE_MODEL),
        help="Local pyannote speaker-diarization-community-1 snapshot",
    )
    parser.add_argument("--pyannote-device", default="cuda")
    parser.add_argument("--pyannote-timeout-s", type=float, default=3600.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build speaker-pure audio clips with frozen Gemini 2.5 Flash ASR"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run or resume the production data pipeline")
    run.add_argument("inputs", nargs="+", help="Video file(s) or directories")
    run.add_argument("--output-dir", required=True)
    run.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    run.add_argument("--resume", action="store_true")
    run.add_argument("--no-master", action="store_true", help="Do not retain the decoded source-rate PCM master")
    run.add_argument("--speaker-turns-dir", default=None, help="Reuse per-video acoustic diarization turns JSON files")
    run.add_argument("--asr-on-speaker-rejected", action="store_true")
    _speaker_runtime_arguments(run)

    # Connectivity can vary by deployment. The ASR model, prompt, seed and
    # decoding parameters are deliberately not exposed as CLI overrides.
    run.add_argument("--gemini-backend", choices=["formal_rest"], default=None)
    run.add_argument("--gemini-location", default=None)
    run.add_argument(
        "--gemini-credentials-path",
        default=os.environ.get("GEMINI_CREDENTIALS_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
    run.add_argument("--gemini-timeout-s", type=float, default=None)
    run.add_argument("--gemini-proxy-url", default=None)
    run.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")

    doctor = subparsers.add_parser("doctor", help="Check runtime prerequisites without processing data")
    _speaker_runtime_arguments(doctor)
    doctor.add_argument("--check-gemini", action="store_true")
    doctor.add_argument(
        "--gemini-credentials-path",
        default=os.environ.get("GEMINI_CREDENTIALS_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
    )
    return parser


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    payload = load_json_config(args.config)
    gemini_payload = dict(payload.get("gemini") or {})

    frozen = {
        "model": "gemini-2.5-flash",
        "backend": "formal_rest",
        "prompt_id": PROMPT_ID,
        "seeds": [101],
        "max_attempts": 1,
        "temperature": 0.0,
        "thinking_budget": 0,
        "top_k": 1,
        "max_output_tokens": 1024,
        "quality_enabled": True,
    }
    for key, expected in frozen.items():
        if key in gemini_payload and gemini_payload[key] != expected:
            raise ValueError(
                f"gemini.{key} is frozen for production: expected {expected!r}, "
                f"got {gemini_payload[key]!r}"
            )

    gemini = GeminiConfig(
        backend=args.gemini_backend or str(gemini_payload.get("backend") or "formal_rest"),
        model="gemini-2.5-flash",
        location=args.gemini_location or str(gemini_payload.get("location") or "us-central1"),
        seeds=(101,),
        max_attempts=1,
        timeout_s=max(30.0, args.gemini_timeout_s or float(gemini_payload.get("timeout_s", 180.0))),
        credentials_path=args.gemini_credentials_path,
        proxy_url=args.gemini_proxy_url or gemini_payload.get("proxy_url"),
        prompt_id=PROMPT_ID,
        temperature=0.0,
        thinking_budget=0,
        top_k=1,
        max_output_tokens=1024,
        quality_enabled=True,
    )

    pyannote: PyannoteConfig | None = None
    if not args.speaker_turns_dir:
        pyannote_python = Path(args.pyannote_python).expanduser()
        pyannote_model = Path(args.pyannote_model).expanduser()
        if not pyannote_python.is_file() or not pyannote_model.is_dir():
            raise ValueError(
                "a valid --pyannote-python and --pyannote-model are required "
                "unless --speaker-turns-dir is supplied"
            )
        pyannote = PyannoteConfig(
            python_executable=str(pyannote_python),
            model_dir=str(pyannote_model),
            device=args.pyannote_device,
            request_timeout_s=max(30.0, float(args.pyannote_timeout_s)),
        )

    return PipelineConfig(
        gemini=gemini,
        pyannote=pyannote,
        analysis_sample_rate=int(payload.get("analysis_sample_rate", 16_000)),
        dataset_sample_rate=int(payload.get("dataset_sample_rate", 24_000)),
        keep_master=not args.no_master,
        resume=bool(args.resume),
        vad=dict(payload.get("vad") or {}),
        items=dict(payload.get("items") or {}),
        speaker_timeline=dict(payload.get("speaker_timeline") or {}),
        speaker_gate=dict(payload.get("speaker_gate") or {}),
        audio_quality=dict(payload.get("audio_quality") or {}),
        background_music=dict(payload.get("background_music") or {}),
        background_music_recovery=dict(payload.get("background_music_recovery") or {}),
        text_quality=dict(payload.get("text_quality") or {}),
        short_policy=dict(payload.get("short_policy") or {}),
        voice_effect_visual=dict(payload.get("voice_effect_visual") or {}),
        speaker_turns_dir=args.speaker_turns_dir,
        asr_on_speaker_rejected=bool(args.asr_on_speaker_rejected),
    )


def _doctor(args: argparse.Namespace) -> int:
    pyannote_python = Path(args.pyannote_python).expanduser().resolve()
    pyannote_model = Path(args.pyannote_model).expanduser().resolve()
    pyannote_available = pyannote_python.is_file() and pyannote_model.is_dir()
    checks: list[dict[str, Any]] = [
        {"name": "python", "ok": sys.version_info >= (3, 10), "value": platform.python_version()},
        {"name": "ffmpeg", "ok": bool(shutil.which("ffmpeg")), "value": shutil.which("ffmpeg") or ""},
        {"name": "ffprobe", "ok": bool(shutil.which("ffprobe")), "value": shutil.which("ffprobe") or ""},
        {"name": "pyannote_python", "ok": pyannote_available, "value": str(pyannote_python)},
        {"name": "pyannote_model", "ok": pyannote_available, "value": str(pyannote_model)},
    ]
    if pyannote_available:
        try:
            probe = subprocess.run(
                [
                    str(pyannote_python),
                    "-c",
                    "import pyannote.audio, torch; print('cuda' if torch.cuda.is_available() else 'cpu')",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            checks.append(
                {
                    "name": "pyannote_import",
                    "ok": probe.returncode == 0,
                    "value": str(probe.stdout or probe.stderr or "").strip()[-500:],
                }
            )
        except Exception as exc:
            checks.append({"name": "pyannote_import", "ok": False, "value": f"{type(exc).__name__}: {exc}"})

    if args.check_gemini:
        credentials_path = Path(args.gemini_credentials_path).expanduser().resolve() if args.gemini_credentials_path else None
        formal_client_ok = False
        formal_value = "not configured"
        try:
            import google.auth  # noqa: F401
            import requests  # noqa: F401

            from .formal_gemini_client import GeminiClient as _FormalClient
            from .formal_gemini_client import GeminiConfig as _FormalConfig

            _FormalClient(
                _FormalConfig.from_env(
                    credentials_path=credentials_path,
                    model="gemini-2.5-flash",
                    location="us-central1",
                    timeout_s=180,
                    retries=1,
                )
            )
            formal_client_ok = True
            formal_value = "formal_rest_topk_ready"
        except Exception as exc:
            formal_value = f"{type(exc).__name__}: {exc}"
        checks.extend(
            [
                {
                    "name": "gemini_client",
                    "ok": formal_client_ok,
                    "value": formal_value,
                },
                {
                    "name": "gemini_auth",
                    "ok": bool(credentials_path and credentials_path.is_file()),
                    "value": "configured" if credentials_path and credentials_path.is_file() else "missing",
                },
            ]
        )

    report = {"ok": all(bool(row["ok"]) for row in checks), "checks": checks}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "doctor":
        raise SystemExit(_doctor(args))
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = _build_config(args)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"invalid configuration: {exc}") from exc
    summary = run_pipeline(args.inputs, args.output_dir, config=config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
