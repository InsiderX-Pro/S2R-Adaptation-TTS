from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.speaker import label_candidates_with_turns  # noqa: E402


class SpeakerProjectionTest(unittest.TestCase):
    def test_marks_material_second_speaker_as_overlap(self) -> None:
        candidates = [{"candidate_index": 0, "start_time": 0, "end_time": 1000, "text": "hello"}]
        turns = [
            {"start_ms": 0, "end_ms": 800, "speaker_id": "A", "confidence": 0.9},
            {"start_ms": 700, "end_ms": 1000, "speaker_id": "B", "confidence": 0.8},
        ]
        result = label_candidates_with_turns(candidates, turns, video_id="video")
        self.assertEqual("video_spk01", result[0]["speaker_id"])
        self.assertTrue(result[0]["overlap_detected"])

    def test_understands_atomic_fusion_overlap_state(self) -> None:
        candidates = [{"candidate_index": 0, "start_time": 100, "end_time": 300, "text": "mixed"}]
        turns = [
            {
                "start_ms": 100,
                "end_ms": 300,
                "speaker_state": "overlap",
                "speaker_ids": ["A", "B"],
            }
        ]
        result = label_candidates_with_turns(candidates, turns, video_id="video")
        self.assertTrue(result[0]["overlap_detected"])
        self.assertNotEqual("unknown", result[0]["speaker_id"])

    def test_missing_turn_confidence_remains_unavailable(self) -> None:
        candidates = [{"candidate_index": 0, "start_time": 0, "end_time": 1000, "text": "hello"}]
        turns = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "A"}]
        result = label_candidates_with_turns(candidates, turns, video_id="video")
        self.assertIsNone(result[0]["speaker_confidence"])
        self.assertEqual("unavailable_in_acoustic_turns", result[0]["speaker_confidence_source"])

    def test_regular_overlap_is_audited_without_tiny_boundary_false_positive(self) -> None:
        candidates = [{"candidate_index": 0, "start_time": 0, "end_time": 1000, "text": "hello"}]
        turns = [{"start_ms": 0, "end_ms": 1000, "speaker_id": "A"}]
        tiny = label_candidates_with_turns(
            candidates,
            turns,
            video_id="video",
            overlap_intervals=[{"start_ms": 980, "end_ms": 1000, "speaker_state": "overlap"}],
        )[0]
        material = label_candidates_with_turns(
            candidates,
            turns,
            video_id="video",
            overlap_intervals=[{"start_ms": 970, "end_ms": 1000, "speaker_state": "overlap"}],
        )[0]
        self.assertEqual(20, tiny["acoustic_overlap_ms"])
        self.assertFalse(tiny["overlap_detected"])
        self.assertEqual(30, material["acoustic_overlap_ms"])
        self.assertTrue(material["overlap_detected"])


if __name__ == "__main__":
    unittest.main()
