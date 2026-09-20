from __future__ import annotations

import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.music_detection import (  # noqa: E402
    AudioSetMusicDetector,
    _audio_windows,
    apply_background_music_quality,
)
from ominivoice_data_pipeline.pipeline import discover_videos  # noqa: E402


class MusicDetectionTest(unittest.TestCase):
    @staticmethod
    def _write_wave(path: Path, sample_count: int) -> None:
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(np.zeros(sample_count, dtype="<i2").tobytes())

    def test_windowing_covers_the_tail_without_duplicate_windows(self) -> None:
        audio = np.zeros(21 * 16_000, dtype=np.float32)
        windows = _audio_windows(audio, 16_000, 10.0, 5.0)
        self.assertEqual([(0, 160_000), (80_000, 240_000), (160_000, 320_000), (176_000, 336_000)], [(a, b) for a, b, _ in windows])

    def test_music_result_blocks_asr_before_gemini(self) -> None:
        items = [
            {
                "item_id": "a",
                "audio_sha256": "hash-a",
                "speaker_pure": True,
                "asr_duration_valid": True,
                "audio_valid": True,
                "asr_eligible": True,
                "audio_quality_flags": [],
                "quality_flags": [],
                "asr_ineligible_reasons": [],
                "eligibility": {},
            }
        ]
        reports = [
            {
                "item_id": "a",
                "audio_sha256": "hash-a",
                "detected": True,
                "max_music_probability": 0.563,
                "reasons": ["background_music_detected"],
            }
        ]
        result = apply_background_music_quality(items, reports)[0]
        self.assertFalse(result["audio_valid"])
        self.assertFalse(result["asr_eligible"])
        self.assertIn("background_music_detected", result["audio_quality_flags"])
        self.assertIn("background_music_detected", result["asr_ineligible_reasons"])

    def test_below_asr_minimum_skips_ast_without_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "micro.wav"
            self._write_wave(audio_path, 272)
            detector = AudioSetMusicDetector()
            detector._load = Mock(side_effect=AssertionError("AST must not load"))

            report = detector.analyze_items(
                [
                    {
                        "item_id": "micro",
                        "audio_path": str(audio_path),
                        "audio_sha256": "hash-micro",
                        "asr_duration_valid": False,
                    }
                ]
            )[0]

            detector._load.assert_not_called()
            self.assertTrue(report["analysis_skipped"])
            self.assertEqual("below_asr_min_duration", report["analysis_skip_reason"])
            self.assertFalse(report["detected"])
            self.assertEqual([], report["windows"])
            self.assertEqual([], report["reasons"])

    def test_malformed_micro_clip_is_skipped_even_if_duration_flag_is_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audio_path = Path(tmp) / "micro.wav"
            self._write_wave(audio_path, 256)
            detector = AudioSetMusicDetector()
            detector._load = Mock(side_effect=AssertionError("AST must not load"))

            report = detector.analyze_items(
                [
                    {
                        "item_id": "micro",
                        "audio_path": str(audio_path),
                        "audio_sha256": "hash-micro",
                        "asr_duration_valid": True,
                    }
                ]
            )[0]

            detector._load.assert_not_called()
            self.assertTrue(report["analysis_skipped"])
            self.assertEqual("waveform_too_short_for_ast", report["analysis_skip_reason"])
            self.assertFalse(report["detected"])

    def test_stale_result_fails_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stale"):
            apply_background_music_quality(
                [{"item_id": "a", "audio_sha256": "new"}],
                [{"item_id": "a", "audio_sha256": "old", "reasons": []}],
            )

    def test_audio_sources_are_discoverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sample.wav").touch()
            (root / "ignore.txt").touch()
            self.assertEqual([root / "sample.wav"], discover_videos([root]))


if __name__ == "__main__":
    unittest.main()
