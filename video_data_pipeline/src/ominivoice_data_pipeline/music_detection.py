from __future__ import annotations

import wave
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_MUSIC_CONFIG: dict[str, Any] = {
    "enabled": True,
    "model": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "model_revision": "f826b80d28226b62986cc218e5cec390b1096902",
    "label": "Music",
    "threshold": 0.20,
    "sample_rate": 16_000,
    "window_seconds": 10.0,
    "hop_seconds": 5.0,
    "batch_size": 8,
    "device": "auto",
    # torchaudio.compliance.kaldi.fbank uses a 25 ms / 400-sample frame at
    # 16 kHz.  Keeping this explicit prevents malformed micro-clips from
    # aborting the whole source video before the normal duration gate can
    # discard them.
    "min_waveform_samples": 400,
}


def _selected_config(config: dict[str, Any] | None) -> dict[str, Any]:
    selected = {**DEFAULT_MUSIC_CONFIG, **dict(config or {})}
    if not 0.0 < float(selected["threshold"]) < 1.0:
        raise ValueError("background_music.threshold must be in (0, 1)")
    if int(selected["sample_rate"]) <= 0:
        raise ValueError("background_music.sample_rate must be positive")
    if float(selected["window_seconds"]) <= 0.0 or float(selected["hop_seconds"]) <= 0.0:
        raise ValueError("background_music window and hop must be positive")
    if int(selected["batch_size"]) <= 0:
        raise ValueError("background_music.batch_size must be positive")
    if int(selected["min_waveform_samples"]) < 2:
        raise ValueError("background_music.min_waveform_samples must be at least 2")
    return selected


def _read_pcm_wave(path: str | Path, target_sample_rate: int) -> np.ndarray:
    source = Path(path).expanduser().resolve()
    with wave.open(str(source), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
        raw = handle.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"AST music detector requires 16-bit PCM WAV: {source}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    if sample_rate != target_sample_rate and len(audio):
        target_length = max(1, round(len(audio) * target_sample_rate / sample_rate))
        old_positions = np.arange(len(audio), dtype=np.float64)
        new_positions = np.arange(target_length, dtype=np.float64) * sample_rate / target_sample_rate
        audio = np.interp(new_positions, old_positions, audio).astype(np.float32)
    return audio


def _audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    hop_seconds: float,
) -> list[tuple[int, int, np.ndarray]]:
    window_samples = max(1, round(window_seconds * sample_rate))
    hop_samples = max(1, round(hop_seconds * sample_rate))
    if len(audio) <= window_samples:
        return [(0, len(audio), audio)]
    starts = list(range(0, len(audio) - window_samples + 1, hop_samples))
    last_start = len(audio) - window_samples
    if starts[-1] != last_start:
        starts.append(last_start)
    return [(start, start + window_samples, audio[start : start + window_samples]) for start in starts]


class AudioSetMusicDetector:
    """Batch AudioSet AST inference with a frozen model revision and auditable scores."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = _selected_config(config)
        self._torch: Any = None
        self._extractor: Any = None
        self._model: Any = None
        self._device: str | None = None
        self._label_id: int | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config["enabled"])

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import ASTForAudioClassification, AutoFeatureExtractor
        except ImportError as exc:
            raise RuntimeError("background-music detection requires torch and transformers") from exc
        configured_device = str(self.config["device"])
        device = "cuda" if configured_device == "auto" and torch.cuda.is_available() else configured_device
        if device == "auto":
            device = "cpu"
        model_name = str(self.config["model"])
        revision = str(self.config["model_revision"])
        extractor = AutoFeatureExtractor.from_pretrained(model_name, revision=revision)
        model = ASTForAudioClassification.from_pretrained(model_name, revision=revision).to(device).eval()
        wanted_label = str(self.config["label"])
        label_ids = [int(index) for index, label in model.config.id2label.items() if str(label) == wanted_label]
        if len(label_ids) != 1:
            raise RuntimeError(f"AST label {wanted_label!r} was not found exactly once")
        self._torch = torch
        self._extractor = extractor
        self._model = model
        self._device = device
        self._label_id = label_ids[0]

    def analyze_items(self, items: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        rows = [dict(item) for item in items]
        if not self.enabled:
            return [
                {
                    "item_id": str(item["item_id"]),
                    "audio_sha256": str(item.get("audio_sha256") or ""),
                    "schema_version": "ominivoice.background-music.v1",
                    "enabled": False,
                    "detected": False,
                    "reasons": [],
                }
                for item in rows
            ]
        sample_rate = int(self.config["sample_rate"])
        window_seconds = float(self.config["window_seconds"])
        hop_seconds = float(self.config["hop_seconds"])
        min_waveform_samples = int(self.config["min_waveform_samples"])
        pending: list[tuple[str, int, int, np.ndarray]] = []
        metadata: dict[str, dict[str, Any]] = {}
        for item in rows:
            item_id = str(item["item_id"])
            audio = _read_pcm_wave(item["audio_path"], sample_rate)
            windows = _audio_windows(audio, sample_rate, window_seconds, hop_seconds)
            metadata[item_id] = {
                "item_id": item_id,
                "audio_sha256": str(item.get("audio_sha256") or ""),
                "duration_ms": round(len(audio) * 1000.0 / sample_rate),
                "windows": [],
                "analysis_skipped": False,
                "analysis_skip_reason": None,
            }
            if not bool(item.get("asr_duration_valid", True)):
                metadata[item_id]["analysis_skipped"] = True
                metadata[item_id]["analysis_skip_reason"] = "below_asr_min_duration"
                continue
            if len(audio) < min_waveform_samples:
                metadata[item_id]["analysis_skipped"] = True
                metadata[item_id]["analysis_skip_reason"] = "waveform_too_short_for_ast"
                continue
            for start, end, samples in windows:
                pending.append((item_id, start, end, samples))

        batch_size = int(self.config["batch_size"])
        if pending:
            self._load()
            assert self._torch is not None and self._extractor is not None and self._model is not None
            assert self._device is not None and self._label_id is not None
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            inputs = self._extractor(
                [samples for _, _, _, samples in batch],
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                scores = self._model(**inputs).logits.sigmoid()[:, self._label_id].detach().cpu().tolist()
            for (item_id, start, end, _), score in zip(batch, scores, strict=True):
                metadata[item_id]["windows"].append(
                    {
                        "start_ms": round(start * 1000.0 / sample_rate),
                        "end_ms": round(end * 1000.0 / sample_rate),
                        "music_probability": round(float(score), 6),
                    }
                )

        threshold = float(self.config["threshold"])
        results: list[dict[str, Any]] = []
        for item in rows:
            data = metadata[str(item["item_id"])]
            maximum = max((float(row["music_probability"]) for row in data["windows"]), default=0.0)
            detected = maximum >= threshold
            results.append(
                {
                    **data,
                    "schema_version": "ominivoice.background-music.v1",
                    "enabled": True,
                    "model": str(self.config["model"]),
                    "model_revision": str(self.config["model_revision"]),
                    "label": str(self.config["label"]),
                    "sample_rate": sample_rate,
                    "window_seconds": window_seconds,
                    "hop_seconds": hop_seconds,
                    "min_waveform_samples": min_waveform_samples,
                    "threshold": threshold,
                    "max_music_probability": round(maximum, 6),
                    "detected": detected,
                    "reasons": ["background_music_detected"] if detected else [],
                }
            )
        return results


def apply_background_music_quality(
    items: list[dict[str, Any]],
    reports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {str(report.get("item_id") or ""): dict(report) for report in reports}
    output: list[dict[str, Any]] = []
    for raw in items:
        item = dict(raw)
        item_id = str(item["item_id"])
        report = by_id.get(item_id)
        if report is None or str(report.get("audio_sha256") or "") != str(item.get("audio_sha256") or ""):
            raise RuntimeError(
                f"missing or stale background-music result for {item_id}: "
                f"item_hash={item.get('audio_sha256')!r}, "
                f"report_hash={(report or {}).get('audio_sha256')!r}, "
                f"available_ids={sorted(by_id)}"
            )
        previous_audio_flags = [str(reason) for reason in list(item.get("audio_quality_flags") or []) if str(reason)]
        music_reasons = [str(reason) for reason in list(report.get("reasons") or []) if str(reason)]
        audio_flags = list(dict.fromkeys(previous_audio_flags + music_reasons))
        audio_valid = bool(item.get("audio_valid", True)) and not music_reasons
        asr_eligible = bool(item.get("speaker_pure")) and bool(item.get("asr_duration_valid")) and audio_valid
        quality_flags = [
            str(reason)
            for reason in list(item.get("quality_flags") or [])
            if str(reason) and str(reason) not in set(previous_audio_flags)
        ]
        ineligible = [
            str(reason)
            for reason in list(item.get("asr_ineligible_reasons") or [])
            if str(reason) and str(reason) not in set(previous_audio_flags)
        ]
        eligibility = dict(item.get("eligibility") or {})
        eligibility.update({"audio_valid": audio_valid, "asr_eligible": asr_eligible, "audio_reasons": audio_flags})
        item.update(
            {
                "local_background_music": report,
                "audio_quality_flags": audio_flags,
                "quality_flags": list(dict.fromkeys(quality_flags + audio_flags)),
                "audio_valid": audio_valid,
                "asr_eligible": asr_eligible,
                "asr_ineligible_reasons": list(dict.fromkeys(ineligible + audio_flags)),
                "eligibility": eligibility,
            }
        )
        output.append(item)
    return output


__all__ = ["AudioSetMusicDetector", "DEFAULT_MUSIC_CONFIG", "apply_background_music_quality"]
