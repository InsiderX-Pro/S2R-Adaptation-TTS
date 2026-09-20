from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.visual_voice_effect import (  # noqa: E402
    apply_identity_protection_review,
    frame_has_identity_protection_card,
)


class VisualVoiceEffectTest(unittest.TestCase):
    def test_identity_protection_card_color_geometry(self) -> None:
        frame = np.full((180, 320, 3), 220, dtype=np.uint8)
        frame[43:137, 20:90] = [5, 20, 90]
        frame[60:120:2, 35:75:2] = [40, 170, 220]
        report = frame_has_identity_protection_card(frame)
        self.assertTrue(report["detected"])

    def test_generic_navy_graphics_outside_card_region_do_not_match(self) -> None:
        frame = np.full((180, 320, 3), 220, dtype=np.uint8)
        frame[20:160, 120:300] = [5, 20, 90]
        report = frame_has_identity_protection_card(frame)
        self.assertFalse(report["detected"])

    def test_full_frame_blue_transition_is_not_an_identity_card(self) -> None:
        frame = np.full((180, 320, 3), [5, 20, 90], dtype=np.uint8)
        frame[43:137, 20:90] = [5, 20, 90]
        frame[43:137:2, 20:90:2] = [40, 170, 220]
        report = frame_has_identity_protection_card(frame)
        self.assertGreater(report["cyan_pixels"], 900)
        self.assertFalse(report["detected"])

    def test_interval_overlap_routes_to_review_before_asr(self) -> None:
        items = [
            {"item_id": "hit", "start_ms": 558_000, "end_ms": 563_000, "asr_eligible": True},
            {"item_id": "miss", "start_ms": 670_000, "end_ms": 675_000, "asr_eligible": True},
        ]
        report = {"intervals": [{"start_ms": 558_500, "end_ms": 632_500}]}
        rows = apply_identity_protection_review(items, report)
        self.assertTrue(rows[0]["voice_effect_review_required"])
        self.assertFalse(rows[0]["asr_eligible"])
        self.assertFalse(rows[0]["eligibility"]["asr_eligible"])
        self.assertIn("voice_effect_review_required", rows[0]["asr_ineligible_reasons"])
        self.assertFalse(rows[1]["voice_effect_review_required"])
        self.assertTrue(rows[1]["asr_eligible"])

    def test_missing_video_stream_routes_every_candidate_to_review(self) -> None:
        rows = apply_identity_protection_review(
            [{"item_id": "audio-only", "start_ms": 1000, "end_ms": 4000, "asr_eligible": True}],
            {"review_all": True, "review_reason": "source_has_no_video_stream", "intervals": []},
        )
        self.assertTrue(rows[0]["voice_effect_review_required"])
        self.assertFalse(rows[0]["asr_eligible"])
        self.assertEqual(["source_has_no_video_stream"], rows[0]["voice_effect_review_reasons"])


if __name__ == "__main__":
    unittest.main()
