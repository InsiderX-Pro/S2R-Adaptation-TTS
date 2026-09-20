from __future__ import annotations

import struct
import sys
import tempfile
import unittest
import wave
import json
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ominivoice_data_pipeline.gemini_asr import GeminiConfig  # noqa: E402
from ominivoice_data_pipeline.pipeline import (  # noqa: E402
    PipelineConfig,
    _build_training_manifest,
    _apply_gate_eligibility,
    _apply_speaker_safe_padding,
    _cap_candidate_durations,
    _finalize_gemini_only,
    _fused_segments_for_export,
    _normalize_speaker_timeline_payload,
    _publish_final_clips,
    _resume_asr_rows,
    _resume_quality_rows,
    _select_asr_attempts,
    _speaker_gate,
    _validate_production_gemini,
)
from ominivoice_data_pipeline.segmentation import prepare_speaker_timelines  # noqa: E402


class PipelineHelpersTest(unittest.TestCase):
    def test_speaker_gate_rejects_unknown_tail(self) -> None:
        candidates = [
            {
                "candidate_index": 0,
                "start_time": 0,
                "end_time": 1000,
                "speaker_id": "v_spk01",
                "speaker_confidence": 1.0,
                "text": "one",
            },
            {
                "candidate_index": 1,
                "start_time": 1100,
                "end_time": 1500,
                "speaker_id": "unknown",
                "speaker_confidence": 0.0,
                "text": "tail",
            },
        ]
        result = _speaker_gate(candidates, config={})
        self.assertTrue(result["segments"][0]["gate"]["passed"])
        self.assertFalse(result["segments"][1]["gate"]["passed"])

    def test_gate_refreshes_asr_eligibility_after_merge(self) -> None:
        rows = _apply_gate_eligibility(
            [
                {
                    "gate": {"passed": False, "reasons": ["overlap_ratio_exceeded"]},
                    "speaker_quality_flags": [],
                    "duration_quality_flags": [],
                    "audio_quality_flags": [],
                    "quality_flags": [],
                    "asr_duration_valid": True,
                    "audio_valid": True,
                    "asr_eligibility_flags": [],
                }
            ]
        )
        self.assertFalse(rows[0]["speaker_pure"])
        self.assertFalse(rows[0]["asr_eligible"])
        self.assertEqual(
            ["speaker_gate_failed", "broadcast_crosstalk_detected"],
            rows[0]["asr_ineligible_reasons"],
        )

    def test_speaker_rejected_override_never_bypasses_duration_or_audio(self) -> None:
        items = [
            {"item_id": "speaker-only", "speaker_pure": False, "asr_duration_valid": True, "audio_valid": True},
            {"item_id": "too-short", "speaker_pure": True, "asr_duration_valid": False, "audio_valid": True},
            {"item_id": "bad-audio", "speaker_pure": True, "asr_duration_valid": True, "audio_valid": False},
        ]
        attempted = _select_asr_attempts(items, include_speaker_rejected=True)
        self.assertEqual(["speaker-only"], [row["item_id"] for row in attempted])

    def test_speaker_rejected_override_never_bypasses_voice_effect_review(self) -> None:
        items = [
            {
                "item_id": "visual-review",
                "speaker_pure": False,
                "asr_duration_valid": True,
                "audio_valid": True,
                "asr_eligible": False,
                "voice_effect_review_required": True,
                "asr_ineligible_reasons": ["speaker_gate_failed", "voice_effect_review_required"],
            }
        ]
        self.assertEqual([], _select_asr_attempts(items, include_speaker_rejected=True))

    def test_overlong_candidate_is_split_without_text_duplication(self) -> None:
        result = _cap_candidate_durations(
            {
                "utterances": [
                    {
                        "candidate_index": 9,
                        "start_time": 1000,
                        "end_time": 26001,
                        "text": "unaligned transcript",
                        "words": [],
                    }
                ]
            },
            max_duration_ms=12_000,
        )
        parts = result["utterances"]
        self.assertEqual(3, len(parts))
        self.assertLessEqual(max(part["end_time"] - part["start_time"] for part in parts), 12_000)
        self.assertTrue(all(part["text"] == "" for part in parts))

    def test_overlong_candidate_prefers_nearby_silence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audio = Path(directory) / "silence-boundary.wav"
            samples = [0 if 7900 <= index < 8100 else 1000 for index in range(14_000)]
            with wave.open(str(audio), "wb") as writer:
                writer.setnchannels(1)
                writer.setsampwidth(2)
                writer.setframerate(1000)
                writer.writeframes(struct.pack(f"<{len(samples)}h", *samples))
            result = _cap_candidate_durations(
                {
                    "utterances": [
                        {
                            "candidate_index": 0,
                            "start_time": 0,
                            "end_time": 14_000,
                            "text": "unaligned",
                            "words": [],
                        }
                    ]
                },
                max_duration_ms=12_000,
                audio_path=audio,
                split_search_radius_ms=1_500,
            )
        self.assertTrue(7_850 <= result["utterances"][0]["end_time"] <= 8_100)

    def test_speaker_safe_padding_stops_at_conflicting_turns(self) -> None:
        padded = _apply_speaker_safe_padding(
            [{"item_id": "a", "start_ms": 1000, "end_ms": 2000, "acoustic_gate_speaker_id": "A"}],
            speaker_timeline=[
                {"start_ms": 900, "end_ms": 950, "speaker_id": "B", "speaker_state": "single"},
                {"start_ms": 950, "end_ms": 2075, "speaker_id": "A", "speaker_state": "single"},
                {"start_ms": 2075, "end_ms": 2200, "speaker_id": "B", "speaker_state": "single"},
            ],
            padding_ms=150,
            audio_duration_ms=3000,
        )[0]
        self.assertEqual((950, 2075), (padded["export_start_ms"], padded["export_end_ms"]))

    def test_sidecar_normalization_round_trips_regular_overlap(self) -> None:
        normalized = _normalize_speaker_timeline_payload(
            {
                "turns": [{"start_ms": 0, "end_ms": 2000, "speaker_id": "A"}],
                "regular_timeline": [
                    {"start_ms": 0, "end_ms": 1200, "speaker_id": "A"},
                    {"start_ms": 800, "end_ms": 2000, "speaker_id": "B"},
                ],
                "overlap_intervals": [
                    {"start_ms": 800, "end_ms": 1200, "speaker_ids": ["A", "B"], "speaker_state": "overlap"}
                ],
            },
            config={},
        )
        prepared = prepare_speaker_timelines(normalized)
        self.assertTrue(prepared["assignment_valid"])
        self.assertEqual(1, prepared["summary"]["overlap_interval_count"])

    def test_fuses_vad_with_atomic_speaker_boundaries(self) -> None:
        rows, fusion = _fused_segments_for_export(
            "video",
            {"segments": [{"start": 0.0, "end": 2.0}]},
            {
                "turns": [
                    {"start": 0.0, "end": 1.2, "speaker_id": "A"},
                    {"start": 0.8, "end": 2.0, "speaker_id": "B"},
                ]
            },
        )
        self.assertEqual(["single", "overlap", "single"], [row["speaker_state"] for row in rows])
        self.assertEqual(3, fusion["summary"]["fused_segment_count"])

    def test_resume_asr_rows_requires_exact_audio_hash_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            from ominivoice_data_pipeline.gemini_asr import PROMPT_ID, PROMPT_SHA256

            path.write_text(
                json.dumps(
                    {
                        "item_id": "a",
                        "audio_sha256": "hash-a",
                        "prompt_id": PROMPT_ID,
                        "prompt_sha256": PROMPT_SHA256,
                        "runs": [{"status": "ok", "transcript": "ဟုတ်"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertIsNotNone(_resume_asr_rows(path, [{"item_id": "a", "audio_sha256": "hash-a"}]))
            self.assertIsNone(_resume_asr_rows(path, [{"item_id": "a", "audio_sha256": "changed"}]))

    def test_resume_asr_rows_retries_recorded_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "item_id": "a",
                        "audio_sha256": "hash-a",
                        "prompt_id": "p3_flash_canonical_greedy",
                        "prompt_sha256": __import__("ominivoice_data_pipeline.gemini_asr", fromlist=["PROMPT_SHA256"]).PROMPT_SHA256,
                        "runs": [{"status": "error", "transcript": "", "error": "ProxyError"}],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertIsNone(_resume_asr_rows(path, [{"item_id": "a", "audio_sha256": "hash-a"}]))

    def test_resume_quality_rows_rejects_stale_prompt_contract(self) -> None:
        from ominivoice_data_pipeline.gemini_asr import QUALITY_PROMPT_ID, QUALITY_PROMPT_SHA256

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "quality.jsonl"
            item = {"item_id": "a", "audio_sha256": "hash-a"}
            primary = {
                "item_id": "a",
                "runs": [{"status": "ok", "transcript": "ဟုတ်"}],
            }
            transcript_hash = __import__("hashlib").sha256("ဟုတ်".encode("utf-8")).hexdigest()
            stale = {
                "item_id": "a",
                "audio_sha256": "hash-a",
                "primary_transcript_sha256": transcript_hash,
                "prompt_id": "q2_flash_audio_text_quality_guard",
                "prompt_sha256": "stale",
            }
            path.write_text(json.dumps(stale) + "\n", encoding="utf-8")
            self.assertIsNone(_resume_quality_rows(path, [item], [primary]))
            current = {
                **stale,
                "prompt_id": QUALITY_PROMPT_ID,
                "prompt_sha256": QUALITY_PROMPT_SHA256,
                "status": "ok",
                "quality_assessment": {"speech_complete": True},
            }
            path.write_text(json.dumps(current) + "\n", encoding="utf-8")
            self.assertIsNotNone(_resume_quality_rows(path, [item], [primary]))

    def test_publish_final_clips_contains_only_accepted_audio(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "candidate_clips"
            candidates.mkdir()
            accepted = candidates / "accepted.wav"
            rejected = candidates / "rejected.wav"
            accepted.write_bytes(b"accepted-audio")
            rejected.write_bytes(b"rejected-audio")
            final = root / "final_clips"
            final.mkdir()
            (final / "stale.wav").write_bytes(b"stale")
            rows = _publish_final_clips(
                [{"item_id": "a", "audio_path": str(accepted), "accepted": True}],
                final,
            )
            self.assertEqual(["accepted.wav"], sorted(path.name for path in final.glob("*.wav")))
            self.assertEqual(str(accepted.resolve()), rows[0]["candidate_audio_path"])
            self.assertEqual(str((final / "accepted.wav").resolve()), rows[0]["audio_path"])
            self.assertNotEqual(str(rejected.resolve()), rows[0]["audio_path"])

    def test_single_gemini_success_is_accepted(self) -> None:
        items = [{"item_id": "a", "duration_ms": 1_200, "asr_eligible": True, "short_answer": False}]
        rows = [
            {
                "item_id": "a",
                "model": "gemini-2.5-flash",
                "prompt_id": "p3_flash_canonical_greedy",
                "runs": [
                    {
                        "status": "ok",
                        "seed": 101,
                        "transcript": "မင်္ဂလာပါ",
                        "guard": {"reasons": []},
                    }
                ],
            }
        ]
        quality_rows = [
            {
                "item_id": "a",
                "status": "ok",
                "prompt_id": "q3_flash_audio_text_voice_disguise_quality_guard",
                "quality_assessment": {
                    "speech_complete": True,
                    "natural_short_response": False,
                    "breath_or_noise_only": False,
                    "background_music": False,
                    "sound_effects": False,
                    "broadcast_crosstalk": False,
                    "voice_disguise_effect": False,
                    "low_snr": False,
                    "clipping": False,
                    "english_natural_code_switch": False,
                    "english_excessive": False,
                    "english_audio_text_consistent": True,
                },
            }
        ]
        finalized = _finalize_gemini_only(items, rows, quality_rows)
        self.assertTrue(finalized[0]["accepted"])
        self.assertEqual("မင်္ဂလာပါ", finalized[0]["final_text"])
        self.assertEqual("gemini_flash", finalized[0]["text_source"])

    def test_nfc_and_quality_rejections_are_applied(self) -> None:
        items = [{"item_id": "a", "duration_ms": 1_200, "asr_eligible": True, "short_answer": False}]
        assessment = {
            "speech_complete": True,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": True,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": True,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": True,
            "english_excessive": False,
            "english_audio_text_consistent": True,
        }
        rows = [{"item_id": "a", "runs": [{"status": "ok", "transcript": "cafe\u0301", "guard": {"reasons": []}}]}]
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertEqual("café", finalized[0]["gemini_text"])
        self.assertFalse(finalized[0]["accepted"])
        self.assertIn("background_music_detected", finalized[0]["quality_rejection_reasons"])
        self.assertIn("voice_disguise_effect_detected", finalized[0]["quality_rejection_reasons"])

    def test_unclear_short_response_is_rejected(self) -> None:
        items = [{"item_id": "a", "duration_ms": 700, "asr_eligible": True, "short_answer": True}]
        assessment = {
            "speech_complete": True,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": False,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": False,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": False,
            "english_excessive": False,
            "english_audio_text_consistent": True,
        }
        rows = [{"item_id": "a", "runs": [{"status": "ok", "transcript": "ဟုတ်", "guard": {"reasons": []}}]}]
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertFalse(finalized[0]["accepted"])
        self.assertIn("unclear_or_unnatural_short_response", finalized[0]["quality_rejection_reasons"])

    def test_training_manifest_targets_three_percent_short(self) -> None:
        accepted = [
            {"item_id": f"long-{index}", "short_answer_accepted": False}
            for index in range(97)
        ] + [
            {"item_id": f"short-{index}", "short_answer_accepted": True}
            for index in range(20)
        ]
        training, summary = _build_training_manifest(accepted)
        self.assertEqual(3, summary["selected_short_count"])
        self.assertEqual(100, len(training))
        self.assertEqual(0.03, summary["actual_short_ratio"])

    def test_gemini_failure_routes_to_retry_but_speaker_rejection_stays_excluded(self) -> None:
        items = [
            {"item_id": "retry", "asr_eligible": True},
            {"item_id": "excluded", "asr_eligible": False, "asr_ineligible_reasons": ["speaker_gate_failed"]},
        ]
        rows = [
            {"item_id": "retry", "runs": [{"status": "error", "seed": 101}]},
            {"item_id": "excluded", "runs": []},
        ]
        finalized = _finalize_gemini_only(items, rows, [])
        self.assertTrue(finalized[0]["retry_required"])
        self.assertFalse(finalized[0]["excluded"])
        self.assertTrue(finalized[1]["excluded"])
        self.assertFalse(finalized[1]["retry_required"])

    def test_quality_call_failure_routes_to_retry_without_rejecting_primary_text(self) -> None:
        items = [{"item_id": "a", "duration_ms": 1_200, "asr_eligible": True}]
        rows = [{"item_id": "a", "runs": [{"status": "ok", "transcript": "ဟုတ်", "guard": {"reasons": []}}]}]
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "error", "quality_assessment": {}}],
        )
        self.assertTrue(finalized[0]["retry_required"])
        self.assertFalse(finalized[0]["excluded"])
        self.assertEqual("ဟုတ်", finalized[0]["gemini_text"])
        self.assertIn("gemini_quality_failed", finalized[0]["qa_reasons"])

    def test_acronym_code_switch_is_warning_not_hard_rejection(self) -> None:
        items = [{"item_id": "a", "duration_ms": 2_000, "asr_eligible": True}]
        rows = [
            {
                "item_id": "a",
                "runs": [{"status": "ok", "transcript": "NUG က သတင်း ထုတ်ပြန် ပါတယ်", "guard": {"reasons": []}}],
            }
        ]
        assessment = {
            "speech_complete": True,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": False,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": False,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": False,
            "english_excessive": True,
            "english_audio_text_consistent": True,
        }
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertTrue(finalized[0]["accepted"])
        self.assertEqual([], finalized[0]["quality_rejection_reasons"])
        self.assertIn("english_code_switch_review", finalized[0]["quality_warning_reasons"])
        self.assertEqual("auto_accept_with_warning", finalized[0]["qa_route"])

    def test_substantial_unnatural_english_remains_hard_rejection(self) -> None:
        items = [{"item_id": "a", "duration_ms": 2_000, "asr_eligible": True}]
        rows = [
            {
                "item_id": "a",
                "runs": [
                    {
                        "status": "ok",
                        "transcript": "official press briefing သတင်း ထုတ်ပြန်ချက်",
                        "guard": {"reasons": []},
                    }
                ],
            }
        ]
        assessment = {
            "speech_complete": True,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": False,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": False,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": False,
            "english_excessive": False,
            "english_audio_text_consistent": True,
        }
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertFalse(finalized[0]["accepted"])
        self.assertIn("unnatural_english", finalized[0]["quality_rejection_reasons"])

    def test_english_audio_text_mismatch_remains_hard_rejection(self) -> None:
        items = [{"item_id": "a", "duration_ms": 2_000, "asr_eligible": True}]
        rows = [{"item_id": "a", "runs": [{"status": "ok", "transcript": "ဟုတ်ပါတယ်", "guard": {"reasons": []}}]}]
        assessment = {
            "speech_complete": True,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": False,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": False,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": False,
            "english_excessive": False,
            "english_audio_text_consistent": False,
        }
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertFalse(finalized[0]["accepted"])
        self.assertIn("english_audio_text_mismatch", finalized[0]["quality_rejection_reasons"])

    def test_incomplete_tail_routes_to_boundary_repair_not_permanent_exclusion(self) -> None:
        items = [{"item_id": "a", "duration_ms": 9_000, "asr_eligible": True}]
        rows = [{"item_id": "a", "runs": [{"status": "ok", "transcript": "စကား မပြီးသေး", "guard": {"reasons": []}}]}]
        assessment = {
            "speech_complete": False,
            "natural_short_response": False,
            "breath_or_noise_only": False,
            "background_music": False,
            "sound_effects": False,
            "broadcast_crosstalk": False,
            "voice_disguise_effect": False,
            "low_snr": False,
            "clipping": False,
            "english_natural_code_switch": False,
            "english_excessive": False,
            "english_audio_text_consistent": True,
        }
        finalized = _finalize_gemini_only(
            items,
            rows,
            [{"item_id": "a", "status": "ok", "quality_assessment": assessment}],
        )
        self.assertTrue(finalized[0]["boundary_repair_required"])
        self.assertFalse(finalized[0]["accepted"])
        self.assertFalse(finalized[0]["excluded"])
        self.assertEqual("boundary_repair", finalized[0]["qa_route"])

    def test_production_gemini_contract_rejects_drift(self) -> None:
        _validate_production_gemini(PipelineConfig())
        with self.assertRaisesRegex(ValueError, "configuration mismatch"):
            _validate_production_gemini(PipelineConfig(gemini=GeminiConfig(seeds=(7,))))


if __name__ == "__main__":
    unittest.main()
