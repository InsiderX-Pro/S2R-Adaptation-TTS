from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

from .io_utils import canonical_json_sha256, read_json, sha256_file, write_json
from .quality import analyze_wav_quality


DEFAULT_RECOVERY_CONFIG: dict[str, Any] = {
    "enabled": False,
    "process_all": False,
    "python_executable": os.environ.get("SEPARATOR_PYTHON", sys.executable),
    "helper_script": str(Path(__file__).with_name("bs_roformer_worker.py")),
    "audio_separator_version": "0.44.5",
    "model_filename": "model_bs_roformer_ep_317_sdr_12.9755.ckpt",
    "model_sha256": "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa",
    "model_config_sha256": "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f",
    "model_file_dir": os.environ.get("AUDIO_SEPARATOR_MODEL_DIR", "models/audio-separator"),
    "minimum_padded_seconds": 12.0,
    "use_autocast": True,
    "max_post_music_probability": 0.10,
    "min_music_probability_reduction": 0.05,
    "min_signal_level_dbfs": -45.0,
    "max_signal_loss_db": 18.0,
    "max_duration_delta_ms": 100,
    "timeout_seconds": 1_200,
}


def _selected_config(config: dict[str, Any] | None) -> dict[str, Any]:
    selected = {**DEFAULT_RECOVERY_CONFIG, **dict(config or {})}
    timeout_override = os.getenv("OMINIVOICE_SEPARATOR_TIMEOUT_SECONDS")
    if timeout_override:
        selected["timeout_seconds"] = int(timeout_override)
    if not 0.0 <= float(selected["max_post_music_probability"]) < 1.0:
        raise ValueError("background_music_recovery.max_post_music_probability must be in [0, 1)")
    if not 0.0 <= float(selected["min_music_probability_reduction"]) < 1.0:
        raise ValueError("background_music_recovery.min_music_probability_reduction must be in [0, 1)")
    if float(selected["minimum_padded_seconds"]) < 10.0:
        raise ValueError("background_music_recovery.minimum_padded_seconds must be at least 10")
    if int(selected["max_duration_delta_ms"]) < 0 or int(selected["timeout_seconds"]) <= 0:
        raise ValueError("background_music_recovery duration delta and timeout are invalid")
    return selected


def _compact_music(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": report.get("model"),
        "model_revision": report.get("model_revision"),
        "threshold": report.get("threshold"),
        "max_music_probability": report.get("max_music_probability"),
        "detected": bool(report.get("detected")),
    }


def _recovery_output_path(output_dir: Path, item: dict[str, Any]) -> Path:
    return output_dir / Path(str(item["audio_path"])).name


def _cache_reusable(
    report: dict[str, Any],
    item: dict[str, Any],
    initial: dict[str, Any],
    config_sha256: str,
) -> bool:
    if (
        str(report.get("item_id") or "") != str(item["item_id"])
        or str(report.get("original_audio_sha256") or "") != str(item.get("audio_sha256") or "")
        or str(report.get("config_sha256") or "") != config_sha256
        or bool(report.get("initial_detected")) != bool(initial.get("detected"))
    ):
        return False
    status = str(report.get("status") or "")
    if status in {"not_needed", "rejected"}:
        return True
    if status != "recovered":
        return False
    recovered = Path(str(report.get("recovered_audio_path") or "")).expanduser().resolve()
    return recovered.is_file() and sha256_file(recovered) == str(report.get("recovered_audio_sha256") or "")


class BackgroundMusicRecovery:
    """Run an isolated BS-RoFormer vocal separator in gated or direct-all mode."""

    def __init__(self, config: dict[str, Any] | None = None, *, dataset_sample_rate: int = 24_000):
        self.config = _selected_config(config)
        self.dataset_sample_rate = int(dataset_sample_rate)
        if self.dataset_sample_rate <= 0:
            raise ValueError("dataset_sample_rate must be positive")
        self.config_sha256 = canonical_json_sha256(self.config)

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    @property
    def process_all(self) -> bool:
        return bool(self.config["process_all"])

    def _run_separator(
        self,
        items: list[dict[str, Any]],
        output_dir: Path,
        request_path: Path,
        response_path: Path,
    ) -> dict[str, dict[str, Any]]:
        request = {
            "schema_version": "ominivoice.bs-roformer-request.v1",
            "audio_separator_version": str(self.config["audio_separator_version"]),
            "model_filename": str(self.config["model_filename"]),
            "model_sha256": str(self.config["model_sha256"]),
            "model_config_sha256": str(self.config["model_config_sha256"]),
            "model_file_dir": str(Path(str(self.config["model_file_dir"])).expanduser().resolve()),
            "dataset_sample_rate": self.dataset_sample_rate,
            "minimum_padded_seconds": float(self.config["minimum_padded_seconds"]),
            "use_autocast": bool(self.config["use_autocast"]),
            "output_dir": str(output_dir),
            "items": [
                {
                    "item_id": str(item["item_id"]),
                    "audio_path": str(item["audio_path"]),
                    "duration_ms": int(item["duration_ms"]),
                    "output_path": str(_recovery_output_path(output_dir, item)),
                }
                for item in items
            ],
        }
        write_json(request_path, request)
        if response_path.is_file():
            response_path.unlink()
        command = [
            str(self.config["python_executable"]),
            str(self.config["helper_script"]),
            "--request",
            str(request_path),
            "--response",
            str(response_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=int(self.config["timeout_seconds"]),
        )
        if completed.returncode or not response_path.is_file():
            detail = (completed.stderr or completed.stdout or "").strip()[-6000:]
            raise RuntimeError(f"BS-RoFormer batch failed ({completed.returncode}): {detail}")
        payload = read_json(response_path)
        return {
            str(row.get("item_id") or ""): dict(row)
            for row in list(payload.get("items") or [])
            if isinstance(row, dict)
        }

    def recover_items(
        self,
        items: Iterable[dict[str, Any]],
        initial_reports: Iterable[dict[str, Any]],
        *,
        output_dir: str | Path,
        music_detector: Any,
        audio_quality_config: dict[str, Any],
        cached_reports: Iterable[dict[str, Any]] = (),
        request_path: str | Path,
        response_path: str | Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        rows = [dict(item) for item in items]
        initial_by_id = {str(report.get("item_id") or ""): dict(report) for report in initial_reports}
        if self.process_all:
            initial_by_id = {
                str(item["item_id"]): {
                    "item_id": str(item["item_id"]),
                    "audio_sha256": str(item.get("audio_sha256") or ""),
                    "schema_version": "ominivoice.background-music.v1",
                    "enabled": False,
                    "detected": False,
                    "reasons": [],
                    "analysis_skipped": True,
                    "analysis_skip_reason": "direct_all_separation",
                }
                for item in rows
            }
        cache_by_id = {str(report.get("item_id") or ""): dict(report) for report in cached_reports}
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        reports: dict[str, dict[str, Any]] = {}
        pending: list[dict[str, Any]] = []

        for item in rows:
            item_id = str(item["item_id"])
            initial = initial_by_id.get(item_id)
            if initial is None or str(initial.get("audio_sha256") or "") != str(item.get("audio_sha256") or ""):
                raise RuntimeError(f"missing or stale initial background-music report for {item_id}")
            cached = cache_by_id.get(item_id)
            if cached is not None and _cache_reusable(cached, item, initial, self.config_sha256):
                reports[item_id] = cached
                continue
            if self.process_all and not self.enabled:
                reports[item_id] = {
                    "schema_version": "ominivoice.background-music-recovery.v1",
                    "item_id": item_id,
                    "original_audio_path": str(item["audio_path"]),
                    "original_audio_sha256": str(item.get("audio_sha256") or ""),
                    "config_sha256": self.config_sha256,
                    "enabled": False,
                    "processing_mode": "direct_all_no_music_probability",
                    "initial_detected": False,
                    "status": "rejected",
                    "recovered": False,
                    "reasons": ["background_music_recovery_disabled"],
                }
            elif self.process_all:
                pending.append(item)
            elif not bool(initial.get("detected")):
                reports[item_id] = {
                    "schema_version": "ominivoice.background-music-recovery.v1",
                    "item_id": item_id,
                    "original_audio_path": str(item["audio_path"]),
                    "original_audio_sha256": str(item.get("audio_sha256") or ""),
                    "config_sha256": self.config_sha256,
                    "enabled": self.enabled,
                    "initial_detected": False,
                    "status": "not_needed",
                    "recovered": False,
                    "reasons": [],
                }
            elif not self.enabled:
                reports[item_id] = {
                    "schema_version": "ominivoice.background-music-recovery.v1",
                    "item_id": item_id,
                    "original_audio_path": str(item["audio_path"]),
                    "original_audio_sha256": str(item.get("audio_sha256") or ""),
                    "config_sha256": self.config_sha256,
                    "enabled": False,
                    "initial_detected": True,
                    "status": "rejected",
                    "recovered": False,
                    "initial_music": _compact_music(initial),
                    "reasons": ["background_music_recovery_disabled"],
                }
            else:
                pending.append(item)

        separator_rows: dict[str, dict[str, Any]] = {}
        if pending:
            try:
                separator_rows = self._run_separator(
                    pending,
                    output,
                    Path(request_path).expanduser().resolve(),
                    Path(response_path).expanduser().resolve(),
                )
            except Exception as exc:
                separator_rows = {
                    str(item["item_id"]): {
                        "item_id": str(item["item_id"]),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for item in pending
                }
        successful_outputs: list[dict[str, Any]] = []
        for item in pending:
            item_id = str(item["item_id"])
            separated = separator_rows.get(item_id, {})
            recovered_path = _recovery_output_path(output, item)
            if separated.get("status") != "ok" or not recovered_path.is_file():
                reports[item_id] = {
                    "schema_version": "ominivoice.background-music-recovery.v1",
                    "item_id": item_id,
                    "original_audio_path": str(item["audio_path"]),
                    "original_audio_sha256": str(item.get("audio_sha256") or ""),
                    "config_sha256": self.config_sha256,
                    "enabled": True,
                    "processing_mode": (
                        "direct_all_no_music_probability" if self.process_all else "ast_probability_gated"
                    ),
                    "initial_detected": bool(initial_by_id[item_id].get("detected")),
                    "status": "error",
                    "recovered": False,
                    "initial_music": None if self.process_all else _compact_music(initial_by_id[item_id]),
                    "reasons": ["background_music_separation_failed"],
                    "error": str(separated.get("error") or "missing separator result"),
                }
                continue
            recovered_sha256 = sha256_file(recovered_path)
            if recovered_sha256 != str(separated.get("output_sha256") or ""):
                raise RuntimeError(f"BS-RoFormer output hash mismatch for {item_id}")
            successful_outputs.append(
                {
                    **item,
                    "audio_path": str(recovered_path),
                    "audio_sha256": recovered_sha256,
                }
            )

        post_music_rows = (
            []
            if self.process_all
            else music_detector.analyze_items(successful_outputs) if successful_outputs else []
        )
        post_music_by_id = {str(row["item_id"]): dict(row) for row in post_music_rows}
        successful_by_id = {str(item["item_id"]): item for item in successful_outputs}
        for item_id, recovered_item in successful_by_id.items():
            initial = initial_by_id[item_id]
            post_music = None if self.process_all else post_music_by_id[item_id]
            post_quality = analyze_wav_quality(recovered_item["audio_path"], audio_quality_config)
            original_quality = next(
                (dict(item.get("local_audio_quality") or {}) for item in rows if str(item["item_id"]) == item_id),
                {},
            )
            initial_probability = None if self.process_all else float(initial.get("max_music_probability") or 0.0)
            post_probability = None if self.process_all else float(post_music.get("max_music_probability") or 0.0)
            probability_reduction = (
                None if self.process_all else float(initial_probability) - float(post_probability)
            )
            post_metrics = dict(post_quality.get("metrics") or {})
            original_metrics = dict(original_quality.get("metrics") or {})
            duration_delta_ms = abs(
                int(post_metrics.get("duration_ms") or 0) - int(recovered_item.get("duration_ms") or 0)
            )
            signal_level = float(post_metrics.get("signal_level_dbfs") or -180.0)
            original_signal = float(original_metrics.get("signal_level_dbfs") or signal_level)
            signal_loss_db = original_signal - signal_level
            reasons = [str(reason) for reason in list(post_quality.get("reasons") or []) if str(reason)]
            if not self.process_all:
                if float(post_probability) > float(self.config["max_post_music_probability"]):
                    reasons.append("background_music_residual")
                if float(probability_reduction) < float(self.config["min_music_probability_reduction"]):
                    reasons.append("insufficient_music_probability_reduction")
            if signal_level < float(self.config["min_signal_level_dbfs"]):
                reasons.append("separated_vocals_too_quiet")
            if signal_loss_db > float(self.config["max_signal_loss_db"]):
                reasons.append("separated_vocals_signal_loss")
            if duration_delta_ms > int(self.config["max_duration_delta_ms"]):
                reasons.append("separated_audio_duration_mismatch")
            reasons = list(dict.fromkeys(reasons))
            recovered = not reasons
            reports[item_id] = {
                "schema_version": "ominivoice.background-music-recovery.v1",
                "item_id": item_id,
                "original_audio_path": next(
                    str(item["audio_path"]) for item in rows if str(item["item_id"]) == item_id
                ),
                "original_audio_sha256": next(
                    str(item.get("audio_sha256") or "") for item in rows if str(item["item_id"]) == item_id
                ),
                "config_sha256": self.config_sha256,
                "enabled": True,
                "processing_mode": (
                    "direct_all_no_music_probability" if self.process_all else "ast_probability_gated"
                ),
                "initial_detected": bool(initial.get("detected")),
                "status": "recovered" if recovered else "rejected",
                "recovered": recovered,
                "model": {
                    "backend": "audio-separator",
                    "audio_separator_version": str(self.config["audio_separator_version"]),
                    "model_filename": str(self.config["model_filename"]),
                    "model_sha256": str(self.config["model_sha256"]),
                    "model_config_sha256": str(self.config["model_config_sha256"]),
                },
                "recovered_audio_path": str(recovered_item["audio_path"]),
                "recovered_audio_sha256": str(recovered_item["audio_sha256"]),
                "initial_music": None if self.process_all else _compact_music(initial),
                "post_separation_music": post_music,
                "music_probability_reduction": (
                    None if probability_reduction is None else round(probability_reduction, 6)
                ),
                "post_audio_quality": post_quality,
                "artifact_metrics": {
                    "duration_delta_ms": duration_delta_ms,
                    "signal_loss_db": round(signal_loss_db, 3),
                },
                "reasons": reasons,
            }

        updated_items: list[dict[str, Any]] = []
        effective_music: list[dict[str, Any]] = []
        ordered_reports: list[dict[str, Any]] = []
        for raw in rows:
            item = dict(raw)
            item_id = str(item["item_id"])
            initial = initial_by_id[item_id]
            report = reports[item_id]
            ordered_reports.append(report)
            item["background_music_recovery"] = report
            if report.get("status") == "recovered":
                original_audio_path = str(item["audio_path"])
                original_audio_sha256 = str(item.get("audio_sha256") or "")
                item["pre_separation_audio_path"] = original_audio_path
                item["pre_separation_audio_sha256"] = original_audio_sha256
                item["audio_path"] = str(report["recovered_audio_path"])
                item["audio_sha256"] = str(report["recovered_audio_sha256"])
                post_quality = dict(report["post_audio_quality"])
                old_audio_flags = set(str(value) for value in list(item.get("audio_quality_flags") or []))
                new_audio_flags = [str(value) for value in list(post_quality.get("reasons") or []) if str(value)]
                item["local_audio_quality"] = post_quality
                item["audio_quality_flags"] = new_audio_flags
                item["quality_flags"] = list(
                    dict.fromkeys(
                        [str(value) for value in list(item.get("quality_flags") or []) if str(value) not in old_audio_flags]
                        + new_audio_flags
                    )
                )
                item["asr_ineligible_reasons"] = list(
                    dict.fromkeys(
                        [
                            str(value)
                            for value in list(item.get("asr_ineligible_reasons") or [])
                            if str(value) not in old_audio_flags
                        ]
                        + new_audio_flags
                    )
                )
                item["audio_valid"] = not new_audio_flags
                item["asr_eligible"] = (
                    bool(item.get("speaker_pure"))
                    and bool(item.get("asr_duration_valid"))
                    and bool(item["audio_valid"])
                )
                eligibility = dict(item.get("eligibility") or {})
                eligibility.update(
                    {
                        "audio_valid": bool(item["audio_valid"]),
                        "asr_eligible": bool(item["asr_eligible"]),
                        "audio_reasons": new_audio_flags,
                    }
                )
                item["eligibility"] = eligibility
                if self.process_all:
                    post = {
                        "item_id": item_id,
                        "audio_sha256": str(item.get("audio_sha256") or ""),
                        "schema_version": "ominivoice.background-music.v1",
                        "enabled": False,
                        "detected": False,
                        "reasons": [],
                        "analysis_skipped": True,
                        "analysis_skip_reason": "direct_all_separation",
                        "recovered_by_separation": True,
                        "recovery_status": "recovered",
                    }
                else:
                    post = dict(report["post_separation_music"])
                    post.update(
                        {
                            "initial_detected": True,
                            "initial_max_music_probability": initial.get("max_music_probability"),
                            "recovered_by_separation": True,
                            "recovery_status": "recovered",
                        }
                    )
                effective_music.append(post)
            else:
                effective = dict(initial)
                effective.update(
                    {
                        "initial_detected": bool(initial.get("detected")),
                        "recovered_by_separation": False,
                        "recovery_status": report.get("status"),
                    }
                )
                effective_music.append(effective)
                if self.process_all:
                    blocking_reasons = [
                        str(reason) for reason in list(report.get("reasons") or []) if str(reason)
                    ] or ["background_music_separation_failed"]
                    item["audio_quality_flags"] = list(
                        dict.fromkeys(list(item.get("audio_quality_flags") or []) + blocking_reasons)
                    )
                    item["quality_flags"] = list(
                        dict.fromkeys(list(item.get("quality_flags") or []) + blocking_reasons)
                    )
                    item["asr_ineligible_reasons"] = list(
                        dict.fromkeys(list(item.get("asr_ineligible_reasons") or []) + blocking_reasons)
                    )
                    item["audio_valid"] = False
                    item["asr_eligible"] = False
                    eligibility = dict(item.get("eligibility") or {})
                    eligibility.update(
                        {
                            "audio_valid": False,
                            "asr_eligible": False,
                            "audio_reasons": blocking_reasons,
                        }
                    )
                    item["eligibility"] = eligibility
            updated_items.append(item)
        return updated_items, effective_music, ordered_reports


__all__ = ["BackgroundMusicRecovery", "DEFAULT_RECOVERY_CONFIG"]
