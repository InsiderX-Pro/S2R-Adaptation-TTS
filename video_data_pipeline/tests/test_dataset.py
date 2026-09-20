from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.dataset import build_dataset_items  # noqa: E402


def candidate(index: int, start: int, end: int, speaker: str, confidence: float = 1.0) -> dict:
    return {
        "candidate_index": index,
        "start_time": start,
        "end_time": end,
        "speaker_id": speaker,
        "speaker_confidence": confidence,
        "text": f"text-{index}",
        "gate": {"passed": True, "reasons": []},
    }


class DatasetTest(unittest.TestCase):
    def test_never_merges_across_speaker_boundary(self) -> None:
        rows = build_dataset_items(
            [
                candidate(0, 0, 1000, "v_spk01"),
                candidate(1, 1100, 2000, "v_spk01"),
                candidate(2, 2050, 3100, "v_spk02"),
            ],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )
        self.assertEqual(2, len(rows))
        self.assertEqual([0, 1], rows[0]["candidate_indices"])
        self.assertEqual("v_spk02", rows[1]["speaker_id"])

    def test_failed_gate_does_not_block_same_speaker_merge(self) -> None:
        first = candidate(0, 0, 1000, "v_spk01")
        second = candidate(1, 1050, 2100, "v_spk01")
        second["gate"] = {"passed": False, "reasons": ["multiple_speakers"]}
        rows = build_dataset_items(
            [first, second],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )
        self.assertEqual(1, len(rows))
        self.assertEqual([0, 1], rows[0]["candidate_indices"])
        self.assertIn("speaker_gate_failed", rows[0]["quality_flags"])
        self.assertFalse(rows[0]["speaker_pure"])
        self.assertTrue(rows[0]["duration_valid"])
        self.assertTrue(rows[0]["audio_valid"])
        self.assertFalse(rows[0]["asr_eligible"])

    def test_short_answer_is_duration_invalid_but_still_asr_eligible(self) -> None:
        rows = build_dataset_items(
            [candidate(0, 0, 500, "v_spk01")],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
            min_duration_ms=1_000,
        )

        row = rows[0]
        self.assertTrue(row["speaker_pure"])
        self.assertTrue(row["speaker_accepted"])
        self.assertFalse(row["duration_valid"])
        self.assertTrue(row["short_answer"])
        self.assertTrue(row["asr_duration_valid"])
        self.assertTrue(row["audio_valid"])
        self.assertTrue(row["asr_eligible"])
        self.assertEqual(row["duration_quality_flags"], ["too_short"])
        self.assertEqual(row["asr_ineligible_reasons"], [])
        self.assertEqual(row["eligibility"]["speaker_reasons"], [])

    def test_clip_below_asr_floor_is_not_eligible(self) -> None:
        row = build_dataset_items(
            [candidate(0, 0, 299, "v_spk01")],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
            min_duration_ms=1_000,
            asr_min_duration_ms=300,
        )[0]

        self.assertTrue(row["speaker_pure"])
        self.assertFalse(row["short_answer"])
        self.assertFalse(row["asr_duration_valid"])
        self.assertFalse(row["asr_eligible"])
        self.assertEqual(row["asr_eligibility_flags"], ["below_asr_min_duration"])
        self.assertEqual(row["asr_ineligible_reasons"], ["below_asr_min_duration"])

    def test_empty_firstpass_text_does_not_claim_audio_is_invalid(self) -> None:
        item = candidate(0, 0, 1_200, "v_spk01")
        item["text"] = ""

        row = build_dataset_items(
            [item],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )[0]

        self.assertIn("empty_firstpass_text", row["quality_flags"])
        self.assertTrue(row["speaker_pure"])
        self.assertTrue(row["duration_valid"])
        self.assertTrue(row["audio_valid"])
        self.assertTrue(row["asr_eligible"])
        self.assertTrue(row["review_required"])

    def test_invalid_audio_span_is_separate_from_speaker_purity(self) -> None:
        row = build_dataset_items(
            [candidate(0, 300, 300, "v_spk01")],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )[0]

        self.assertTrue(row["speaker_pure"])
        self.assertFalse(row["duration_valid"])
        self.assertFalse(row["audio_valid"])
        self.assertFalse(row["asr_eligible"])
        self.assertEqual(row["audio_quality_flags"], ["invalid_audio_span"])

    def test_explicit_candidate_audio_invalidity_blocks_asr(self) -> None:
        item = candidate(0, 0, 1_200, "v_spk01")
        item["audio_valid"] = False

        row = build_dataset_items(
            [item],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )[0]

        self.assertTrue(row["speaker_accepted"])
        self.assertFalse(row["audio_valid"])
        self.assertFalse(row["asr_eligible"])

    def test_missing_speaker_confidence_is_unknown_not_low_confidence(self) -> None:
        item = candidate(0, 0, 1_200, "v_spk01")
        item.pop("speaker_confidence")

        row = build_dataset_items(
            [item],
            source_video="video.mp4",
            source_sha256="a" * 64,
            video_id="v",
        )[0]

        self.assertIsNone(row["speaker_confidence"])
        self.assertFalse(row["speaker_confidence_available"])
        self.assertNotIn("low_speaker_confidence", row["speaker_quality_flags"])
        self.assertTrue(row["speaker_pure"])
        self.assertTrue(row["asr_eligible"])


if __name__ == "__main__":
    unittest.main()
