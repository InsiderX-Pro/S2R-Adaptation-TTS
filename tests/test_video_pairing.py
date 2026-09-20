import json
from pathlib import Path
import tempfile
import unittest

from s2r_adaptation.video import prepare_pairs
from s2r_adaptation.manifests import build_manifest, read_rows


class VideoPairingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "video.jsonl"

    def tearDown(self):
        self.temporary.cleanup()

    def row(self, key, video="recording-A", **changes):
        return {"item_id": key, "video_id": video, "speaker_id": "SPEAKER_00",
            "audio_path": f"audio/{key}.wav", "audio_sha256": key * 64,
            "final_text": "abcd!", "duration_ms": 2000, "accepted": True,
            "speaker_pure": True, **changes}

    def write(self, rows):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in rows))

    def test_reference_stays_in_recording_and_text_is_preserved(self):
        self.write([self.row("a"), self.row("b"), self.row("c", "recording-B"), self.row("d", "recording-B")])
        report = prepare_pairs(self.source, self.root / "pairs")
        result = {row["id"]: row for row in read_rows(self.root / "pairs/primary.jsonl")}
        self.assertEqual(report["paired_rows"], 4)
        self.assertEqual(result["a"]["prompt_source_id"], "b")
        self.assertEqual(result["c"]["prompt_source_id"], "d")
        self.assertNotEqual(result["a"]["speaker_id"], result["c"]["speaker_id"])
        self.assertEqual(result["a"]["text"], "abcd!")

    def test_pairing_is_stable_under_input_reordering(self):
        rows = [self.row(key) for key in "abcd"]
        results = []
        for index, values in enumerate([rows, list(reversed(rows))]):
            self.write(values)
            prepare_pairs(self.source, self.root / str(index), seed=17)
            results.append({row["id"]: row["prompt_source_id"] for row in read_rows(self.root / str(index) / "primary.jsonl")})
        self.assertEqual(*results)

    def test_duplicate_audio_cannot_be_its_own_reference(self):
        self.write([self.row("a"), self.row("b", audio_sha256="a" * 64)])
        with self.assertRaisesRegex(ValueError, "no eligible"):
            prepare_pairs(self.source, self.root / "pairs")
        self.assertFalse((self.root / "pairs").exists())

    def test_excluded_clips_and_unpaired_groups_are_reported(self):
        self.write([self.row("a"), self.row("b"), self.row("c", accepted=False),
                    self.row("d", duration_ms=21000), self.row("e", "other-recording")])
        report = prepare_pairs(self.source, self.root / "pairs")
        self.assertEqual(report["paired_rows"], 2)
        self.assertEqual(report["excluded_rows"], 3)

    def test_video_pairs_feed_cubic_manifest_without_field_rewriting(self):
        self.write([self.row("a"), self.row("b")])
        prepare_pairs(self.source, self.root / "pairs")
        secondary = self.root / "secondary.jsonl"
        secondary.write_text(json.dumps({"id":"b", "hypothesis":"", "status":"success", "audio_sha256":"b" * 64}) + "\n" +
            json.dumps({"id":"a", "hypothesis":"ab", "status":"success", "audio_sha256":"a" * 64}) + "\n")
        build_manifest(self.root / "pairs/primary.jsonl", self.root / "weighted",
                       secondary=secondary, require_audio_sha=True)
        rows = list(read_rows(self.root / "weighted/train.jsonl"))
        self.assertEqual([row["pseudo_label_weight"] for row in rows], [.125, .1])
        self.assertEqual(rows[0]["prompt_source_id"], "b")
        self.assertEqual(rows[0]["text"], "abcd!")


if __name__ == "__main__": unittest.main()
