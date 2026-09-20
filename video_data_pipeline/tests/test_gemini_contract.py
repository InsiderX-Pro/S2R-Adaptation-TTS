from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.formal_gemini_client import (  # noqa: E402
    GeminiClient as FormalGeminiClient,
)
from ominivoice_data_pipeline.formal_gemini_client import (  # noqa: E402
    GeminiConfig as FormalGeminiConfig,
)
from ominivoice_data_pipeline.gemini_asr import (  # noqa: E402
    PROMPT_ID,
    PROMPT_SHA256,
    QUALITY_PROMPT_ID,
)


class FrozenGeminiContractTest(unittest.TestCase):
    def test_primary_prompt_is_the_validated_p3_prompt(self) -> None:
        self.assertEqual("p3_flash_canonical_greedy", PROMPT_ID)
        self.assertEqual(
            "21cd22664334bcafc77a177119d8fd061366f6bee64e84b9b78132a4767dc6fb",
            PROMPT_SHA256,
        )
        self.assertNotEqual(PROMPT_ID, QUALITY_PROMPT_ID)

    def test_formal_rest_payload_writes_top_k_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            credentials = root / "service-account.json"
            credentials.write_text(json.dumps({"project_id": "test-project"}), encoding="utf-8")
            audio = root / "clip.wav"
            audio.write_bytes(b"RIFFtest")
            client = FormalGeminiClient(
                FormalGeminiConfig(credentials_path=credentials, retries=1, retry_base_s=0.0)
            )
            response_payload = {
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
            }
            with mock.patch.object(client, "_generate_once", return_value=response_payload) as generate_once:
                client.generate_content(
                    "frozen prompt",
                    media_path=audio,
                    top_k=1,
                    seed=101,
                    temperature=0.0,
                    thinking_budget=0,
                    max_output_tokens=1024,
                    retries=1,
                )
            request_payload = generate_once.call_args.args[0]
            self.assertEqual(1, request_payload["generationConfig"]["topK"])
            self.assertEqual(101, request_payload["generationConfig"]["seed"])
            self.assertEqual(0.0, request_payload["generationConfig"]["temperature"])
            self.assertEqual(
                {"thinkingBudget": 0},
                request_payload["generationConfig"]["thinkingConfig"],
            )


if __name__ == "__main__":
    unittest.main()
