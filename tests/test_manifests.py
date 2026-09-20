import json
from pathlib import Path
import tempfile
import unittest

from s2r_adaptation.manifests import build_manifest, read_rows
from s2r_adaptation.firered import stage_config, verify_manifest


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.primary = self.root / "primary.jsonl"
        self.secondary = self.root / "asr2.jsonl"
        self.primary_rows = [{"id": "b", "text": "abcd!", "spoken_text": "abcd!", "audio_sha256": "x",
                              "prompt_audio_path": "ref.wav", "loss_weight": .5},
                             {"id": "a", "text": "Aé", "audio_sha256": "y"}]
        self.secondary_rows = [{"id": "a", "hypothesis": "", "status": "success", "audio_sha256": "y"},
                               {"id": "b", "hypothesis": "ab", "status": "success", "audio_sha256": "x"}]
        self.write(self.primary, self.primary_rows)
        self.write(self.secondary, self.secondary_rows)

    def tearDown(self): self.tmp.cleanup()

    def write(self, path, rows):
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    def build(self, **kwargs):
        return build_manifest(self.primary, self.root / "out", secondary=self.secondary, **kwargs)

    def test_join_order_text_and_reference_preserved(self):
        report = self.build(require_audio_sha=True)
        rows = list(read_rows(self.root / "out/train.jsonl"))
        self.assertEqual([r["id"] for r in rows], ["b", "a"])
        self.assertEqual(rows[0]["text"], "abcd!")
        self.assertEqual(rows[0]["prompt_audio_path"], "ref.wav")
        self.assertEqual(rows[0]["pseudo_label_weight"], .125)
        self.assertEqual(rows[0]["loss_weight"], .125)
        self.assertEqual(rows[1]["pseudo_label_weight"], .1)
        self.assertEqual(report["accepted_rows"], 2)
        verify_manifest(self.root / "out/train.jsonl")

    def test_failed_asr_does_not_become_empty_success(self):
        self.secondary_rows[0]["status"] = "failed"
        self.write(self.secondary, self.secondary_rows)
        with self.assertRaisesRegex(ValueError, "secondary_asr_failed"): self.build()
        self.assertFalse((self.root / "out").exists())
        report = self.build(on_asr_error="exclude")
        self.assertEqual(report["excluded_rows"], 1)

    def test_missing_request_fails(self):
        self.write(self.secondary, self.secondary_rows[1:])
        with self.assertRaisesRegex(ValueError, "secondary_asr_missing"): self.build()

    def test_duplicate_ids_fail(self):
        self.write(self.secondary, self.secondary_rows * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"): self.build()
        self.write(self.secondary, self.secondary_rows)
        self.write(self.primary, self.primary_rows * 2)
        with self.assertRaisesRegex(ValueError, "duplicate"): self.build()

    def test_audio_and_primary_binding(self):
        self.secondary_rows[0]["audio_sha256"] = "wrong"
        self.write(self.secondary, self.secondary_rows)
        with self.assertRaisesRegex(ValueError, "audio_sha256_mismatch"): self.build()
        self.secondary_rows[0]["audio_sha256"] = "y"
        self.secondary_rows[0]["reference"] = "changed"
        self.write(self.secondary, self.secondary_rows)
        with self.assertRaisesRegex(ValueError, "primary_transcript_binding_mismatch"): self.build()

    def test_archived_cer_recomputed(self):
        for row in self.secondary_rows:
            row.pop("status")
            row["canonical"] = {"cer": 1 if row["id"] == "a" else .5}
        self.write(self.secondary, self.secondary_rows)
        with self.assertRaisesRegex(ValueError, "status_missing"): self.build()
        self.secondary_rows[0]["canonical"]["cer"] = .9
        self.write(self.secondary, self.secondary_rows)
        with self.assertRaisesRegex(ValueError, "recomputed"): self.build(asr_format="archived")

    def test_empty_primary_excluded(self):
        self.primary_rows.append({"id": "empty", "text": "!?"})
        self.write(self.primary, self.primary_rows)
        self.assertEqual(self.build()["exclusion_reasons"], {"empty_normalized_primary": 1})

    def test_no_overwrite_and_tamper_detection(self):
        self.build()
        with self.assertRaises(FileExistsError): self.build()
        path = self.root / "out/train.jsonl"
        with path.open("a") as out: out.write("\n")
        with self.assertRaisesRegex(ValueError, "checksum"): verify_manifest(path)

    def test_s_stage_unit_weight_and_paper_recipe(self):
        build_manifest(self.primary, self.root / "out", stage="S")
        config = stage_config({}, self.root / "out/train.jsonl", self.root / "train", self.root / "base")
        self.assertEqual(config["max_steps"], 53640)
        self.assertEqual(config["learning_rate"], 2e-6)
        self.assertIsNone(config["train_sources"][0]["loss_weight"])
        self.assertIsNone(config["resume_from"])
        self.assertEqual([r["pseudo_label_weight"] for r in read_rows(self.root / "out/train.jsonl")], [1, 1])

    def test_r_stage_fresh_optimizer_and_model_only_initialization(self):
        self.build()
        s = self.root / "S"
        s.mkdir()
        (s / "model.safetensors").write_bytes(b"fixture")
        (s / "config.json").write_text("{}")
        (s / "export_metadata.json").write_text(json.dumps({"status":"passed",
            "checkpoint_type":"full_fsdp_export", "source_checkpoint":"S/checkpoint-00053640"}))
        config = stage_config({"resume_from": "old", "init_adapter_from": "old"},
            self.root / "out/train.jsonl", self.root / "train", self.root / "base", s_core=s)
        self.assertEqual(config["max_steps"], 16015)
        self.assertEqual(config["gradient_accumulation_steps"], 3)
        self.assertEqual(config["learning_rate"], 1e-6)
        self.assertIsNone(config["resume_from"])
        self.assertIsNone(config["init_adapter_from"])
        self.assertEqual(config["core_model_dir"], str(s.resolve()))


if __name__ == "__main__": unittest.main()
