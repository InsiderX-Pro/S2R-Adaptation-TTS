from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.background_music_recovery import BackgroundMusicRecovery  # noqa: E402
from ominivoice_data_pipeline.io_utils import sha256_file  # noqa: E402


def _write_wave(path: Path, *, frames: int = 12_000, amplitude: int = 8_000) -> None:
    samples = bytearray()
    for index in range(frames):
        value = amplitude if (index // 100) % 2 == 0 else -amplitude
        samples.extend(int(value).to_bytes(2, byteorder="little", signed=True))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24_000)
        handle.writeframes(bytes(samples))


class _FakeDetector:
    def __init__(self, probability: float):
        self.probability = probability

    def analyze_items(self, items):
        return [
            {
                "item_id": item["item_id"],
                "audio_sha256": item["audio_sha256"],
                "schema_version": "ominivoice.background-music.v1",
                "model": "fake-ast",
                "model_revision": "fixed",
                "threshold": 0.2,
                "max_music_probability": self.probability,
                "detected": self.probability >= 0.2,
                "reasons": ["background_music_detected"] if self.probability >= 0.2 else [],
            }
            for item in items
        ]


class _FailDetector:
    def analyze_items(self, _items):
        raise AssertionError("music probability detector must not run in process_all mode")


class _FakeRecovery(BackgroundMusicRecovery):
    def __init__(self, template: Path, **kwargs):
        super().__init__(**kwargs)
        self.template = template

    def _run_separator(self, items, output_dir, request_path, response_path):
        results = {}
        for item in items:
            target = output_dir / Path(item["audio_path"]).name
            shutil.copy2(self.template, target)
            results[item["item_id"]] = {
                "item_id": item["item_id"],
                "status": "ok",
                "output_path": str(target),
                "output_sha256": sha256_file(target),
            }
        return results


class BackgroundMusicRecoveryTest(unittest.TestCase):
    def test_runtime_timeout_override_supports_shared_gpu_workers(self) -> None:
        with patch.dict(os.environ, {"OMINIVOICE_SEPARATOR_TIMEOUT_SECONDS": "3600"}):
            recovery = BackgroundMusicRecovery(config={"timeout_seconds": 1200})
        self.assertEqual(3600, recovery.config["timeout_seconds"])

    def _item_and_initial(self, original: Path):
        item = {
            "item_id": "clip-a",
            "audio_path": str(original),
            "audio_sha256": sha256_file(original),
            "duration_ms": 500,
            "speaker_pure": True,
            "asr_duration_valid": True,
            "audio_valid": True,
            "asr_eligible": True,
            "audio_quality_flags": [],
            "quality_flags": [],
            "asr_ineligible_reasons": [],
            "eligibility": {},
            "local_audio_quality": {"metrics": {"signal_level_dbfs": -12.0}},
        }
        initial = {
            "item_id": "clip-a",
            "audio_sha256": item["audio_sha256"],
            "model": "fake-ast",
            "model_revision": "fixed",
            "threshold": 0.2,
            "max_music_probability": 0.5,
            "detected": True,
            "reasons": ["background_music_detected"],
        }
        return item, initial

    def test_successful_recovery_replaces_audio_and_clears_music_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_raw:
            temporary = Path(temporary_raw)
            original = temporary / "original.wav"
            separated = temporary / "separated.wav"
            _write_wave(original)
            _write_wave(separated, amplitude=6_000)
            item, initial = self._item_and_initial(original)
            recovery = _FakeRecovery(
                separated,
                config={"enabled": True},
                dataset_sample_rate=24_000,
            )
            updated, effective, reports = recovery.recover_items(
                [item],
                [initial],
                output_dir=temporary / "recovered",
                music_detector=_FakeDetector(0.02),
                audio_quality_config={},
                request_path=temporary / "request.json",
                response_path=temporary / "response.json",
            )
            self.assertEqual("recovered", reports[0]["status"])
            self.assertTrue(reports[0]["recovered"])
            self.assertEqual(reports[0]["recovered_audio_path"], updated[0]["audio_path"])
            self.assertEqual(item["audio_path"], updated[0]["pre_separation_audio_path"])
            self.assertFalse(effective[0]["detected"])
            self.assertEqual([], effective[0]["reasons"])
            self.assertTrue(updated[0]["asr_eligible"])

    def test_process_all_skips_probability_detector_and_separates_clean_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_raw:
            temporary = Path(temporary_raw)
            original = temporary / "original.wav"
            separated = temporary / "separated.wav"
            _write_wave(original)
            _write_wave(separated, amplitude=6_000)
            item, _initial = self._item_and_initial(original)
            recovery = _FakeRecovery(
                separated,
                config={"enabled": True, "process_all": True},
                dataset_sample_rate=24_000,
            )
            updated, effective, reports = recovery.recover_items(
                [item],
                [],
                output_dir=temporary / "recovered",
                music_detector=_FailDetector(),
                audio_quality_config={},
                request_path=temporary / "request.json",
                response_path=temporary / "response.json",
            )
            self.assertEqual("recovered", reports[0]["status"])
            self.assertEqual("direct_all_no_music_probability", reports[0]["processing_mode"])
            self.assertIsNone(reports[0]["initial_music"])
            self.assertIsNone(reports[0]["post_separation_music"])
            self.assertIsNone(reports[0]["music_probability_reduction"])
            self.assertEqual(reports[0]["recovered_audio_path"], updated[0]["audio_path"])
            self.assertTrue(updated[0]["asr_eligible"])
            self.assertFalse(effective[0]["detected"])
            self.assertEqual("direct_all_separation", effective[0]["analysis_skip_reason"])

    def test_residual_music_fails_closed_and_keeps_original_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_raw:
            temporary = Path(temporary_raw)
            original = temporary / "original.wav"
            separated = temporary / "separated.wav"
            _write_wave(original)
            _write_wave(separated)
            item, initial = self._item_and_initial(original)
            recovery = _FakeRecovery(
                separated,
                config={"enabled": True},
                dataset_sample_rate=24_000,
            )
            updated, effective, reports = recovery.recover_items(
                [item],
                [initial],
                output_dir=temporary / "recovered",
                music_detector=_FakeDetector(0.15),
                audio_quality_config={},
                request_path=temporary / "request.json",
                response_path=temporary / "response.json",
            )
            self.assertEqual("rejected", reports[0]["status"])
            self.assertIn("background_music_residual", reports[0]["reasons"])
            self.assertEqual(item["audio_path"], updated[0]["audio_path"])
            self.assertTrue(effective[0]["detected"])
            self.assertIn("background_music_detected", effective[0]["reasons"])

    def test_disabled_recovery_rejects_detected_item_without_separator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_raw:
            temporary = Path(temporary_raw)
            original = temporary / "original.wav"
            _write_wave(original)
            item, initial = self._item_and_initial(original)
            recovery = BackgroundMusicRecovery({"enabled": False})
            updated, effective, reports = recovery.recover_items(
                [item],
                [initial],
                output_dir=temporary / "recovered",
                music_detector=_FakeDetector(0.0),
                audio_quality_config={},
                request_path=temporary / "request.json",
                response_path=temporary / "response.json",
            )
            self.assertEqual("rejected", reports[0]["status"])
            self.assertEqual(["background_music_recovery_disabled"], reports[0]["reasons"])
            self.assertEqual(item["audio_path"], updated[0]["audio_path"])
            self.assertTrue(effective[0]["detected"])


if __name__ == "__main__":
    unittest.main()
