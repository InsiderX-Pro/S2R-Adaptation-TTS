from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.cli import DEFAULT_CONFIG_PATH, _build_config, build_parser  # noqa: E402
from ominivoice_data_pipeline.gemini_asr import PROMPT_ID  # noqa: E402


class CliProductionConfigTest(unittest.TestCase):
    def test_build_config_uses_frozen_flash_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "run",
                "video.mp4",
                "--output-dir",
                "run",
                "--config",
                str(DEFAULT_CONFIG_PATH),
                "--speaker-turns-dir",
                "turns",
            ]
        )
        config = _build_config(args)
        self.assertIsNone(config.pyannote)
        self.assertEqual("gemini-2.5-flash", config.gemini.model)
        self.assertEqual("formal_rest", config.gemini.backend)
        self.assertEqual(PROMPT_ID, config.gemini.prompt_id)
        self.assertEqual((101,), config.gemini.seeds)
        self.assertEqual(0.0, config.gemini.temperature)
        self.assertEqual(0, config.gemini.thinking_budget)
        self.assertEqual(1, config.gemini.top_k)
        self.assertEqual(1024, config.gemini.max_output_tokens)
        self.assertTrue(config.gemini.quality_enabled)

    def test_rejects_scientific_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
            source["gemini"]["seeds"] = [7]
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(source), encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "run",
                    "video.mp4",
                    "--output-dir",
                    "run",
                    "--config",
                    str(config_path),
                    "--speaker-turns-dir",
                    "turns",
                ]
            )
            with self.assertRaisesRegex(ValueError, "gemini.seeds is frozen"):
                _build_config(args)

    def test_packaged_and_operator_default_configs_match(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        operator_payload = json.loads((project_root / "configs" / "default.json").read_text(encoding="utf-8"))
        packaged_payload = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        self.assertEqual(operator_payload, packaged_payload)
        self.assertEqual(15000, packaged_payload["items"]["max_duration_ms"])
        self.assertEqual(12000, packaged_payload["vad"]["target_segment_ms"])
        self.assertEqual(350, packaged_payload["items"]["audio_padding_right_ms"])
        self.assertEqual([101], packaged_payload["gemini"]["seeds"])
        self.assertNotIn("speaker_review", packaged_payload)
        self.assertNotIn("asr_decision", packaged_payload)


if __name__ == "__main__":
    unittest.main()
