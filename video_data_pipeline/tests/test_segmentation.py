from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
import wave


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "ominivoice_data_pipeline"
    / "segmentation.py"
)
SPEC = importlib.util.spec_from_file_location("ominivoice_segmentation_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
SEGMENTATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SEGMENTATION)
energy_vad = SEGMENTATION.energy_vad
find_quiet_boundary_ms = SEGMENTATION.find_quiet_boundary_ms
clean_speaker_timeline = SEGMENTATION.clean_speaker_timeline
fuse_timelines = SEGMENTATION.fuse_timelines
final_single_speaker_gate = SEGMENTATION.final_single_speaker_gate
prepare_speaker_timelines = SEGMENTATION.prepare_speaker_timelines


def _write_pcm16(
    path: Path,
    sections: list[tuple[float, float]],
    *,
    sample_rate: int = 8_000,
    channels: int = 1,
) -> None:
    samples: list[int] = []
    sample_index = 0
    for seconds, amplitude in sections:
        count = int(round(seconds * sample_rate))
        for _ in range(count):
            value = int(
                round(
                    amplitude
                    * 32767.0
                    * math.sin(2.0 * math.pi * 220.0 * sample_index / sample_rate)
                )
            )
            samples.extend([value] * channels)
            sample_index += 1
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(struct.pack(f"<{len(samples)}h", *samples))


class SegmentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.tmp_path = Path(self._temporary_directory.name)

    def test_energy_vad_finds_two_bursts_without_merging_silence(self) -> None:
        audio_path = self.tmp_path / "bursts.wav"
        _write_pcm16(
            audio_path,
            [
                (0.4, 0.0),
                (0.6, 0.40),
                (0.4, 0.0),
                (0.7, 0.30),
                (0.3, 0.0),
            ],
        )

        result = energy_vad(
            {
                "audio_path": str(audio_path),
                "config": {
                    "threshold_dbfs": -35.0,
                    "frame_ms": 20.0,
                    "hop_ms": 10.0,
                    "min_speech_ms": 100.0,
                    "min_silence_ms": 200.0,
                    "speech_pad_ms": 0.0,
                    "merge_gap_ms": 50.0,
                },
            }
        )

        self.assertEqual(result["schema_version"], "ominivoice.energy-vad.v1")
        self.assertEqual(result["audio"]["channels"], 1)
        self.assertEqual(result["analysis"]["threshold_dbfs"], -35.0)
        self.assertEqual(len(result["segments"]), 2)
        self.assertAlmostEqual(result["segments"][0]["start"], 0.4, delta=0.04)
        self.assertAlmostEqual(result["segments"][0]["end"], 1.0, delta=0.04)
        self.assertAlmostEqual(result["segments"][0]["start_time"], 400, delta=40)
        self.assertAlmostEqual(result["segments"][0]["end_time"], 1_000, delta=40)
        self.assertEqual(result["segments"][0]["start_ms"], result["segments"][0]["start_time"])
        self.assertEqual(result["segments"][0]["end_ms"], result["segments"][0]["end_time"])
        self.assertAlmostEqual(result["segments"][1]["start"], 1.4, delta=0.04)
        self.assertAlmostEqual(result["segments"][1]["end"], 2.1, delta=0.04)
        self.assertFalse(result["segments"][0]["is_last"])
        self.assertTrue(result["segments"][1]["is_last"])
        json.dumps(result)

    def test_energy_vad_adaptive_threshold_detects_quiet_speech(self) -> None:
        audio_path = self.tmp_path / "quiet.wav"
        _write_pcm16(audio_path, [(0.5, 0.0), (0.8, 0.025), (0.5, 0.0)])

        result = energy_vad(
            {
                "audio_path": str(audio_path),
                "config": {
                    "min_speech_ms": 100.0,
                    "min_silence_ms": 200.0,
                    "speech_pad_ms": 0.0,
                    "merge_gap_ms": 0.0,
                },
            }
        )

        self.assertAlmostEqual(result["analysis"]["threshold_dbfs"], -55.0)
        self.assertEqual(len(result["segments"]), 1)
        self.assertGreater(result["segments"][0]["duration"], 0.7)

    def test_energy_vad_splits_a_long_continuous_region(self) -> None:
        audio_path = self.tmp_path / "long.wav"
        _write_pcm16(audio_path, [(3.0, 0.3)])

        result = energy_vad(
            {
                "audio_path": str(audio_path),
                "config": {
                    "threshold_dbfs": -35.0,
                    "min_speech_ms": 50.0,
                    "min_silence_ms": 100.0,
                    "speech_pad_ms": 0.0,
                    "merge_gap_ms": 0.0,
                    "max_segment_ms": 1_000.0,
                    "split_search_ms": 0.0,
                    "min_split_segment_ms": 100.0,
                },
            }
        )

        self.assertEqual(len(result["segments"]), 3)
        self.assertTrue(all(segment["duration"] <= 1.001 for segment in result["segments"]))
        self.assertAlmostEqual(result["segments"][0]["start"], 0.0)
        self.assertAlmostEqual(result["segments"][-1]["end"], 3.0)

    def test_energy_vad_prefers_sustained_pause_over_single_quiet_frame(self) -> None:
        audio_path = self.tmp_path / "pause-aware.wav"
        _write_pcm16(
            audio_path,
            [(1.9, 0.3), (0.03, 0.0), (0.55, 0.3), (0.22, 0.0), (1.3, 0.3)],
        )
        result = energy_vad(
            {
                "audio_path": str(audio_path),
                "config": {
                    "threshold_dbfs": -35.0,
                    "min_speech_ms": 50.0,
                    "min_silence_ms": 300.0,
                    "speech_pad_ms": 0.0,
                    "merge_gap_ms": 0.0,
                    "target_segment_ms": 2_000.0,
                    "max_segment_ms": 3_000.0,
                    "split_search_ms": 1_000.0,
                    "min_split_segment_ms": 500.0,
                    "split_pause_window_ms": 180.0,
                    "split_distance_penalty_db": 0.0,
                },
            }
        )
        self.assertEqual(2, len(result["segments"]))
        self.assertTrue(2.42 <= result["segments"][0]["end"] <= 2.68)

    def test_quiet_boundary_prefers_sustained_pause_over_single_quiet_frame(self) -> None:
        audio_path = self.tmp_path / "bounded-pause-aware.wav"
        _write_pcm16(
            audio_path,
            [(1.9, 0.3), (0.03, 0.0), (0.55, 0.3), (0.22, 0.0), (1.3, 0.3)],
        )
        result = find_quiet_boundary_ms(
            audio_path,
            target_ms=2_000,
            lower_ms=1_500,
            upper_ms=3_000,
            pause_window_ms=180.0,
            distance_penalty_db=0.0,
        )
        self.assertEqual("sustained_pause_rms", result["method"])
        self.assertTrue(2_420 <= result["split_ms"] <= 2_680)

    def test_energy_vad_reads_stereo_pcm_and_reports_original_layout(self) -> None:
        audio_path = self.tmp_path / "stereo.wav"
        _write_pcm16(audio_path, [(0.3, 0.0), (0.5, 0.4), (0.3, 0.0)], channels=2)

        result = energy_vad(
            {
                "audio_path": str(audio_path),
                "config": {
                    "threshold_dbfs": -35.0,
                    "min_speech_ms": 50.0,
                    "min_silence_ms": 100.0,
                    "speech_pad_ms": 0.0,
                    "merge_gap_ms": 0.0,
                },
            }
        )

        self.assertEqual(result["audio"]["channels"], 2)
        self.assertEqual(len(result["segments"]), 1)

    def test_fuse_timelines_splits_single_overlap_and_single_regions(self) -> None:
        result = fuse_timelines(
            {
                "vad_segments": [{"id": "vad-1", "start": 0.0, "end": 10.0}],
                "speaker_segments": [
                    {"id": "a1", "start": 0.0, "end": 4.0, "speaker": "A"},
                    {"id": "b1", "start": 4.0, "end": 7.0, "speaker": "B"},
                    {"id": "a2", "start": 6.0, "end": 10.0, "speaker": "A"},
                ],
                "config": {
                    "boundary_tolerance_ms": 0.0,
                    "merge_adjacent_ms": 0.0,
                    "cleanup_enabled": False,
                },
            }
        )

        self.assertEqual(
            [segment["speaker_state"] for segment in result["segments"]],
            ["single", "single", "overlap", "single"],
        )
        self.assertEqual(
            [(segment["start"], segment["end"]) for segment in result["segments"]],
            [(0.0, 4.0), (4.0, 6.0), (6.0, 7.0), (7.0, 10.0)],
        )
        self.assertEqual(result["segments"][0]["speaker_ids"], ["A"])
        self.assertEqual(result["segments"][1]["speaker_ids"], ["B"])
        self.assertEqual(result["segments"][2]["speaker_ids"], ["A", "B"])
        self.assertEqual(result["segments"][2]["speaker"], "OVERLAP")
        self.assertTrue(result["segments"][-1]["is_last"])
        json.dumps(result)

    def test_fuse_timelines_keeps_unassigned_edges_unknown(self) -> None:
        result = fuse_timelines(
            {
                "base_segments": [{"id": "vad-1", "start_time": 0, "end_time": 2_000}],
                "speaker_timeline": [
                    {"start_time": 200, "end_time": 1_800, "speaker_id": "A"}
                ],
                "config": {
                    "boundary_tolerance_ms": 0.0,
                    "merge_adjacent_ms": 0.0,
                    "cleanup_enabled": False,
                },
            }
        )

        self.assertEqual(
            [segment["speaker_state"] for segment in result["segments"]],
            ["unknown", "single", "unknown"],
        )
        self.assertEqual(result["segments"][0]["speaker"], "UNKNOWN")
        self.assertEqual(result["segments"][-1]["speaker"], "UNKNOWN")
        self.assertEqual(result["segments"][1]["start_time"], 200)
        self.assertEqual(result["segments"][1]["end_time"], 1_800)
        self.assertEqual(result["segments"][1]["start_ms"], 200)
        self.assertEqual(result["segments"][1]["end_ms"], 1_800)

    def test_fuse_timelines_absorbs_bounded_unknown_edges_by_default(self) -> None:
        result = fuse_timelines(
            {
                "base_segments": [{"id": "vad-1", "start_ms": 0, "end_ms": 2_000}],
                "turns": [{"start_ms": 200, "end_ms": 1_800, "speaker_id": "A"}],
                "config": {"boundary_tolerance_ms": 0.0, "merge_adjacent_ms": 0.0},
            }
        )

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["speaker_ids"], ["A"])
        self.assertEqual((result["segments"][0]["start_ms"], result["segments"][0]["end_ms"]), (0, 2_000))
        self.assertEqual(result["summary"]["cleanup_action_count"], 2)

    def test_clean_speaker_timeline_merges_same_speaker_gap_and_edges(self) -> None:
        result = clean_speaker_timeline(
            {
                "timeline_start_ms": 0,
                "timeline_end_ms": 2_000,
                "turns": [
                    {"start_ms": 150, "end_ms": 900, "speaker_id": "A"},
                    {"start_ms": 1_050, "end_ms": 1_850, "speaker_id": "A"},
                ],
                "config": {"boundary_tolerance_ms": 0.0, "merge_adjacent_ms": 0.0},
            }
        )

        self.assertEqual(len(result["segments"]), 1)
        self.assertEqual(result["segments"][0]["speaker_id"], "A")
        self.assertEqual(result["segments"][0]["duration_ms"], 2_000)
        self.assertEqual(
            [action["action"] for action in result["cleanup_actions"]],
            ["absorb_unknown_boundary", "absorb_same_speaker_gap", "absorb_unknown_boundary"],
        )

    def test_clean_speaker_timeline_preserves_true_overlap(self) -> None:
        result = clean_speaker_timeline(
            {
                "turns": [
                    {"start_ms": 0, "end_ms": 1_200, "speaker_id": "A"},
                    {"start_ms": 1_000, "end_ms": 2_000, "speaker_id": "B"},
                ],
                "config": {"boundary_tolerance_ms": 0.0, "merge_adjacent_ms": 0.0},
            }
        )

        self.assertEqual(
            [(row["speaker_state"], row["start_ms"], row["end_ms"]) for row in result["segments"]],
            [("single", 0, 1_000), ("overlap", 1_000, 1_200), ("single", 1_200, 2_000)],
        )
        self.assertEqual(result["segments"][1]["speaker_ids"], ["A", "B"])

    def test_prepare_speaker_timelines_separates_assignment_from_overlap(self) -> None:
        result = prepare_speaker_timelines(
            {
                "turns": [
                    {"start_ms": 0, "end_ms": 1_200, "speaker_id": "A"},
                    {"start_ms": 1_000, "end_ms": 2_000, "speaker_id": "B"},
                ],
                "exclusive_turns": [
                    {"start_ms": 0, "end_ms": 1_100, "speaker_id": "A"},
                    {"start_ms": 1_100, "end_ms": 2_000, "speaker_id": "B"},
                ],
                "config": {"boundary_tolerance_ms": 0.0, "merge_adjacent_ms": 0.0},
            }
        )

        self.assertTrue(result["assignment_valid"])
        self.assertEqual(
            [(row["speaker_id"], row["start_ms"], row["end_ms"]) for row in result["assignment_timeline"]],
            [("A", 0, 1_100), ("B", 1_100, 2_000)],
        )
        self.assertEqual(
            [(row["start_ms"], row["end_ms"], row["speaker_ids"]) for row in result["overlap_intervals"]],
            [(1_000, 1_200, ["A", "B"])],
        )
        self.assertEqual(result["summary"]["overlap_interval_count"], 1)

    def test_final_gate_accepts_pure_segments_and_short_tail_silence(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [
                    {"id": "first", "start": 0.0, "end": 1.0},
                    {"id": "last", "start": 1.0, "end": 4.0},
                ],
                "speaker_segments": [
                    {"start": 0.0, "end": 1.0, "speaker": "A"},
                    {"start": 1.0, "end": 3.9, "speaker": "B"},
                ],
            }
        )

        self.assertEqual(
            result["summary"],
            {"total": 2, "accepted": 2, "rejected": 0, "tail_segment_id": "last"},
        )
        self.assertEqual([segment["speaker_id"] for segment in result["accepted"]], ["A", "B"])
        self.assertAlmostEqual(result["segments"][1]["gate"]["trailing_unknown_ms"], 50.0)
        json.dumps(result)

    def test_final_gate_rejects_material_second_speaker(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start": 0.0, "end": 4.0}],
                "speaker_segments": [
                    {"start": 0.0, "end": 3.8, "speaker": "A"},
                    {"start": 3.8, "end": 4.0, "speaker": "B"},
                ],
                "config": {"boundary_collar_ms": 0.0},
            }
        )

        reasons = result["rejected"][0]["gate"]["reasons"]
        self.assertIn("multiple_speakers", reasons)
        self.assertIn("tail_other_speaker_ratio_exceeded", reasons)
        self.assertIn("tail_last_voice_not_dominant", reasons)

    def test_final_gate_tolerates_second_speaker_at_100ms(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start_ms": 0, "end_ms": 5_000}],
                "turns": [
                    {"start_ms": 0, "end_ms": 4_900, "speaker_id": "A"},
                    {"start_ms": 4_900, "end_ms": 5_000, "speaker_id": "B"},
                ],
                "config": {"boundary_collar_ms": 0.0},
            }
        )

        self.assertEqual(result["summary"]["accepted"], 1)
        gate = result["accepted"][0]["gate"]
        self.assertEqual(gate["distinct_speakers"], ["A", "B"])
        self.assertEqual(gate["effective_distinct_speaker_count"], 1)
        self.assertEqual(gate["tolerated_other_speakers"], ["B"])
        self.assertNotIn("multiple_speakers", gate["reasons"])

    def test_final_gate_tolerates_second_speaker_at_two_percent(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start_ms": 0, "end_ms": 10_000}],
                "turns": [
                    {"start_ms": 0, "end_ms": 9_850, "speaker_id": "A"},
                    {"start_ms": 9_850, "end_ms": 10_000, "speaker_id": "B"},
                ],
                "config": {"boundary_collar_ms": 0.0},
            }
        )

        self.assertEqual(result["summary"]["accepted"], 1)
        self.assertEqual(result["accepted"][0]["gate"]["tolerated_other_speakers"], ["B"])

    def test_final_gate_boundary_collar_ignores_edge_only_speaker_jitter(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start": 0.0, "end": 2.0}],
                "turns": [
                    {"start": 0.0, "end": 0.15, "speaker_id": "B"},
                    {"start": 0.15, "end": 1.85, "speaker_id": "A"},
                    {"start": 1.85, "end": 2.0, "speaker_id": "B"},
                ],
                "config": {"boundary_collar_ms": 200.0},
            }
        )

        self.assertEqual(result["summary"]["accepted"], 1)
        gate = result["accepted"][0]["gate"]
        self.assertEqual(gate["distinct_speakers"], ["A"])
        self.assertEqual(gate["boundary_collar_ms_applied"], 200.0)

    def test_final_gate_rejects_long_unassigned_tail_without_inheriting_label(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start": 0.0, "end": 3.0}],
                "speaker_segments": [{"start": 0.0, "end": 2.7, "speaker": "A"}],
            }
        )

        gate = result["rejected"][0]["gate"]
        self.assertAlmostEqual(gate["trailing_unknown_ms"], 250.0)
        self.assertIn("tail_trailing_unknown_exceeded", gate["reasons"])
        self.assertIn("tail_unknown_ratio_exceeded", gate["reasons"])

    def test_final_gate_rejects_tail_with_no_explicit_speaker_evidence(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [{"id": "tail", "start_ms": 0, "end_ms": 1_000}],
                "speaker_segments": [],
            }
        )

        gate = result["rejected"][0]["gate"]
        self.assertIsNone(result["rejected"][0]["speaker_id"])
        self.assertIn("tail_no_explicit_speaker", gate["reasons"])
        self.assertIn("tail_last_voice_not_dominant", gate["reasons"])

    def test_final_gate_selects_tail_by_time_not_input_order(self) -> None:
        result = final_single_speaker_gate(
            {
                "segments": [
                    {"id": "late", "start": 2.0, "end": 3.0},
                    {"id": "early", "start": 0.0, "end": 1.0},
                ],
                "speaker_segments": [
                    {"start": 0.0, "end": 1.0, "speaker": "A"},
                    {"start": 2.0, "end": 3.0, "speaker": "B"},
                ],
            }
        )

        self.assertEqual(result["summary"]["tail_segment_id"], "late")
        self.assertTrue(result["segments"][0]["is_tail"])
        self.assertFalse(result["segments"][1]["is_tail"])
        self.assertEqual(result["summary"]["accepted"], 2)

    def test_rejects_unknown_configuration_keys(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown fusion config keys"):
            fuse_timelines(
                {
                    "segments": [],
                    "speaker_segments": [],
                    "config": {"silently_ignored_typo": True},
                }
            )


if __name__ == "__main__":
    unittest.main()
