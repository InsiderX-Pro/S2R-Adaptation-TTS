"""Frozen RedAE/CAMPPlus feature extraction with an optional disk cache."""

from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch
import torchaudio

from .data import SpeechRecord


class FrozenAudioFeatures:
    def __init__(
        self,
        redae: torch.nn.Module | None,
        speaker_extractor: torch.nn.Module | None,
        *,
        device: torch.device,
        patch_size: int,
        cache_dir: str | Path | None = None,
        cache_metadata: Mapping[str, int] | None = None,
    ):
        if (redae is None) != (speaker_extractor is None):
            raise ValueError("redae and speaker_extractor must both be present or both be absent")
        self.cache_only = redae is None
        if self.cache_only:
            if cache_metadata is None:
                raise ValueError("cache-only features require cache_metadata")
            self.redae = None
            self.speaker_extractor = None
            self.sample_rate = int(cache_metadata["sample_rate"])
            self.downsample_rate = int(cache_metadata["downsample_rate"])
            self.hidden_size = int(cache_metadata["hidden_size"])
        else:
            assert redae is not None and speaker_extractor is not None
            self.redae = redae.eval()
            self.speaker_extractor = speaker_extractor.eval()
            self.sample_rate = int(redae.sample_rate)
            self.downsample_rate = int(redae.downsample_rate)
            self.hidden_size = int(redae.hidden_size)
        self.device = device
        self.patch_size = int(patch_size)
        if not self.cache_only:
            for module in (self.redae, self.speaker_extractor):
                assert module is not None
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_only and self.cache_dir is None:
            raise ValueError("cache-only features require cache_dir")
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, record: SpeechRecord) -> str:
        if record.audio_sha256:
            identity = f"sha256:{record.audio_sha256}"
        else:
            path = Path(record.audio_path)
            stat = path.stat()
            identity = f"path:{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        version = (
            f"redae_sr={self.sample_rate};down={self.downsample_rate};"
            f"hidden={self.hidden_size};tts_patch={self.patch_size}"
        )
        return hashlib.sha256(f"{version}|{identity}".encode("utf-8")).hexdigest()

    def _cache_path(self, record: SpeechRecord) -> Path | None:
        if self.cache_dir is None:
            return None
        key = self._cache_key(record)
        return self.cache_dir / key[:2] / f"{key}.pt"

    @staticmethod
    def _load_audio(path: str) -> tuple[torch.Tensor, int]:
        try:
            audio, sample_rate = torchaudio.load(path)
        except ImportError as error:
            # torchaudio 2.9+ delegates decoding to the optional TorchCodec
            # package.  The training corpora are PCM WAV, so soundfile is a
            # deterministic fallback when that optional decoder is absent.
            if "TorchCodec" not in str(error):
                raise
            import soundfile as sf

            samples, sample_rate = sf.read(
                path,
                dtype="float32",
                always_2d=True,
            )
            audio = torch.from_numpy(samples.T.copy())
        if audio.ndim != 2 or audio.numel() == 0:
            raise ValueError(f"invalid audio tensor from {path}: {tuple(audio.shape)}")
        audio = audio.float().mean(dim=0, keepdim=True)
        if not torch.isfinite(audio).all():
            raise ValueError(f"audio contains NaN/Inf: {path}")
        return audio, int(sample_rate)

    @torch.no_grad()
    def _compute(self, record: SpeechRecord) -> dict[str, torch.Tensor | int | str]:
        if self.cache_only:
            raise FileNotFoundError(
                f"feature cache miss in cache-only mode for {record.record_id}: {record.audio_path}"
            )
        assert self.redae is not None and self.speaker_extractor is not None
        audio, sample_rate = self._load_audio(record.audio_path)
        # Match FireRedTTS3Base.generate exactly: resample first, then left-pad
        # waveform samples so RedAE emits a whole number of TTS patches.
        if sample_rate != self.redae.sample_rate:
            audio = torchaudio.functional.resample(audio, sample_rate, self.redae.sample_rate)
            sample_rate = int(self.redae.sample_rate)
        audio = self.redae.pad_to_multiple_of(
            audio,
            self.redae.downsample_rate * self.patch_size,
        )
        latents = self.redae.encode(audio.to(self.device), sample_rate).float().cpu()
        speaker = self.speaker_extractor(audio, sample_rate).float().cpu()
        return {
            "latents": latents.to(torch.float16),
            "speaker": speaker,
            "sample_rate": sample_rate,
            "audio_path": record.audio_path,
        }

    def _atomic_save(self, value: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        torch.save(value, temporary)
        # Multiple ranks may compute the same prompt.  Atomic replacement keeps
        # every visible cache entry complete; identical writers are harmless.
        os.replace(temporary, path)

    def get(self, record: SpeechRecord) -> tuple[torch.Tensor, torch.Tensor]:
        cache_path = self._cache_path(record)
        value: dict[str, Any]
        if cache_path is not None and cache_path.is_file():
            value = torch.load(cache_path, map_location="cpu", weights_only=True)
        else:
            value = self._compute(record)
            if cache_path is not None:
                self._atomic_save(value, cache_path)
        latents = value["latents"].to(self.device, dtype=torch.float32, non_blocking=True)
        speaker = value["speaker"].to(self.device, dtype=torch.float32, non_blocking=True)
        return latents, speaker
