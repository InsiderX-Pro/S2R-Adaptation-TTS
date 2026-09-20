from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.quality import (  # noqa: E402
    analyze_wav_quality,
    normalize_transcript,
    text_quality_report,
)


def write_wav(path: Path, samples: list[int], sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class TextQualityTest(unittest.TestCase):
    def test_normalizes_nfc_and_whitespace(self) -> None:
        self.assertEqual("café ok", normalize_transcript("  cafe\u0301\t ok  "))

    def test_rejects_empty_repetition_and_abnormal_symbols(self) -> None:
        self.assertIn("empty_text", text_quality_report(" \n ")["reasons"])
        repeated = text_quality_report("hello hello hello hello")
        self.assertIn("repetitive_text", repeated["reasons"])
        abnormal = text_quality_report("မင်္ဂလာပါ 😀")
        self.assertIn("abnormal_characters", abnormal["reasons"])

    def test_allows_natural_scale_code_switch_but_flags_excessive_latin(self) -> None:
        mixed = text_quality_report("ဒီနေ့ meeting ရှိတယ်")
        self.assertNotIn("excessive_english", mixed["reasons"])
        english = text_quality_report("this is almost entirely english စကား")
        self.assertIn("excessive_english", english["reasons"])


class AudioQualityTest(unittest.TestCase):
    def test_detects_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clipped.wav"
            write_wav(path, [32_767] * 32_000)
            report = analyze_wav_quality(path)
        self.assertIn("clipping_detected", report["reasons"])

    def test_detects_low_dynamic_snr_proxy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "low-snr.wav"
            samples = [500 if index % 2 else -500 for index in range(32_000)]
            write_wav(path, samples)
            report = analyze_wav_quality(path)
        self.assertIn("low_snr_detected", report["reasons"])

    def test_accepts_clean_dynamic_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "clean.wav"
            samples: list[int] = []
            for frame in range(100):
                amplitude = 400 if frame < 25 else 8_000
                samples.extend(amplitude if index % 2 else -amplitude for index in range(480))
            write_wav(path, samples)
            report = analyze_wav_quality(path)
        self.assertEqual([], report["reasons"])
        self.assertGreater(report["metrics"]["estimated_snr_db"], 8.0)


if __name__ == "__main__":
    unittest.main()
