import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
import soundfile as sf

from fireredtts3.training.data import (
    DistributedWeightedSampler,
    PromptPairedDataset,
    SpeechRecord,
    SourceSpec,
    load_source_records,
    split_records_by_group,
)
from fireredtts3.training.lora import (
    LoRAConfig,
    build_backbone_route_mask,
    inject_lora,
    set_lora_route_mask,
    trainable_parameter_counts,
)
from fireredtts3.training.features import FrozenAudioFeatures
import fireredtts3.training.trainer as trainer_module
from fireredtts3.training.trainer import TrainerConfig
from fireredtts3.training.model import (
    balanced_stop_loss,
    build_condition_windows,
    build_latent_history_windows,
)


class TrainingDataTest(unittest.TestCase):
    def _manifest(self, root: Path) -> Path:
        path = root / "data.jsonl"
        rows = [
            {"id": "a", "audio_path": str(root / "a.wav"), "text": "A", "speaker_id": "s1", "duration": 2, "video_id": "v1"},
            {"id": "b", "audio_path": str(root / "b.wav"), "text": "B", "speaker_id": "s1", "duration": 3, "video_id": "v1"},
            {"id": "c", "audio_path": str(root / "c.wav"), "text": "C", "speaker_id": "s2", "duration": 2, "video_id": "v2"},
            {"id": "d", "audio_path": str(root / "d.wav"), "text": "D", "speaker_id": "s2", "duration": 3, "video_id": "v2"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_pairing_is_distinct_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = load_source_records(
                SourceSpec(str(self._manifest(root)), "toy"),
                check_audio_exists=False,
            )
            dataset = PromptPairedDataset(records, seed=7)
            first = dataset[0]
            self.assertNotEqual(first["target"].record_id, first["prompt"].record_id)
            self.assertEqual(first["prompt"].record_id, dataset[0]["prompt"].record_id)
            self.assertAlmostEqual(sum(item.sampling_mass for item in dataset.records), 1.0)

    def test_synthetic_voice_is_accepted_as_speaker_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "synthetic.jsonl"
            rows = [
                {
                    "id": "syn-a",
                    "audio_path": str(root / "a.wav"),
                    "spoken_text": "A",
                    "voice": "my-MM-NilarNeural",
                    "audio_duration": 2.0,
                },
                {
                    "id": "syn-b",
                    "audio_path": str(root / "b.wav"),
                    "spoken_text": "B",
                    "voice": "my-MM-NilarNeural",
                    "audio_duration": 2.0,
                },
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            records = load_source_records(
                SourceSpec(str(manifest), "synthetic"),
                check_audio_exists=False,
            )
            self.assertEqual([record.speaker_id for record in records], [
                "my-MM-NilarNeural",
                "my-MM-NilarNeural",
            ])
            paired = PromptPairedDataset(records, seed=42)
            self.assertNotEqual(
                paired[0]["target"].record_id,
                paired[0]["prompt"].record_id,
            )

    def test_pipeline_final_text_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pipeline.jsonl"
            row = {
                "id": "pipeline-a",
                "audio_path": str(root / "a.wav"),
                "final_text": "မြန်မာ စာ",
                "speaker_id": "s1",
                "duration": 2.0,
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            records = load_source_records(
                SourceSpec(str(manifest), "pipeline"),
                check_audio_exists=False,
            )
            self.assertEqual(records[0].text, "မြန်မာ စာ")

    def test_explicit_prompt_preserves_its_own_language(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "crosslingual.jsonl"
            row = {
                "id": "target-my",
                "audio_path": str(root / "target.wav"),
                "text": "မြန်မာစာ",
                "speaker_id": "s1",
                "language_id": "my",
                "prompt_audio_path": str(root / "prompt.wav"),
                "prompt_text": "English prompt",
                "prompt_language_id": "en",
                "duration": 2.0,
            }
            manifest.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
            records = load_source_records(
                SourceSpec(str(manifest), "crosslingual"),
                check_audio_exists=False,
            )
            paired = PromptPairedDataset(records, seed=42)
            self.assertEqual(paired[0]["target"].language_id, "my")
            self.assertEqual(paired[0]["prompt"].language_id, "en")

    def test_prompt_target_text_format_marks_both_languages(self):
        class RecordingTokenizer:
            def __init__(self):
                self.content = None

            def __call__(self, content, **kwargs):
                del kwargs
                self.content = content
                return {"input_ids": [1, 2, 3]}

        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            text_sequence_format="prompt_target_tags",
            prompt_language_tags={"my": "Burmese", "en": "English"},
        )
        config.validate()
        tokenizer = RecordingTokenizer()
        token_ids = trainer_module._build_text_tokens(
            tokenizer,
            config,
            "English prompt",
            "မြန်မာစာ",
            torch.device("cpu"),
            prompt_language_id="en",
        )
        self.assertEqual(tuple(token_ids.shape), (1, 3))
        self.assertEqual(
            tokenizer.content,
            "<|English|><|sot|>English prompt <|Burmese|>မြန်မာစာ<|eot|>",
        )

    def test_pcm_wav_loader_returns_channel_first_float(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "audio.wav"
            samples = torch.linspace(-0.25, 0.25, 1600).numpy()
            sf.write(path, samples, 16000, subtype="PCM_16")
            audio, sample_rate = FrozenAudioFeatures._load_audio(str(path))
            self.assertEqual(sample_rate, 16000)
            self.assertEqual(tuple(audio.shape), (1, 1600))
            self.assertEqual(audio.dtype, torch.float32)
            self.assertTrue(torch.isfinite(audio).all())

    def test_group_split_has_no_video_leakage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = load_source_records(
                SourceSpec(str(self._manifest(root)), "toy"),
                check_audio_exists=False,
            )
            # Search a seed producing one group on each side for this tiny set.
            for seed in range(100):
                try:
                    train, valid = split_records_by_group(records, 0.5, seed=seed)
                    break
                except ValueError:
                    continue
            else:
                self.fail("could not produce a non-empty deterministic split")
            self.assertTrue({x.video_id for x in train}.isdisjoint({x.video_id for x in valid}))

    def test_weighted_sampler_is_rank_partitioned_and_repeatable(self):
        rank0 = DistributedWeightedSampler([0.9, 0.1], num_replicas=2, rank=0, samples_per_epoch=20, seed=3)
        rank1 = DistributedWeightedSampler([0.9, 0.1], num_replicas=2, rank=1, samples_per_epoch=20, seed=3)
        first0, first1 = list(rank0), list(rank1)
        self.assertEqual(first0, list(rank0))
        self.assertEqual(len(first0), 10)
        self.assertEqual(len(first1), 10)

    def test_resume_cli_supersedes_stage_adapter_initializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps({
                    "pretrained_model_dir": "/model",
                    "output_dir": "/output",
                    "train_sources": [{"manifest": "/data.jsonl"}],
                    "init_adapter_from": "/stage-zero",
                    "stage_parent_checkpoint": "/stage-zero",
                }),
                encoding="utf-8",
            )
            with patch.object(trainer_module, "run_training") as run_training:
                trainer_module.main([
                    "--config", str(config_path),
                    "--resume-from", "/checkpoint-2",
                ])
            resolved = run_training.call_args.args[0]
            self.assertEqual(resolved.resume_from, "/checkpoint-2")
            self.assertIsNone(resolved.init_adapter_from)
            self.assertEqual(resolved.stage_parent_checkpoint, "/stage-zero")

    def test_stop_at_step_does_not_mutate_planned_max_steps(self):
        with tempfile.TemporaryDirectory() as temporary:
            config_path = Path(temporary) / "config.json"
            config_path.write_text(
                json.dumps({
                    "pretrained_model_dir": "/model",
                    "output_dir": "/output",
                    "train_sources": [{"manifest": "/data.jsonl"}],
                    "max_steps": 53640,
                    "scheduler_steps": 53640,
                }),
                encoding="utf-8",
            )
            with patch.object(trainer_module, "run_training") as run_training:
                trainer_module.main([
                    "--config", str(config_path),
                    "--stop-at-step", "200",
                ])
            resolved = run_training.call_args.args[0]
            self.assertEqual(resolved.max_steps, 53640)
            self.assertEqual(resolved.scheduler_steps, 53640)
            self.assertEqual(run_training.call_args.kwargs["stop_at_step"], 200)

    def test_training_contract_allows_only_launch_location_to_change(self):
        first = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run-a",
            train_sources=[{"manifest": "data.jsonl"}],
            max_steps=10,
        )
        second = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run-b",
            train_sources=[{"manifest": "data.jsonl"}],
            max_steps=10,
            resume_from="checkpoint",
        )
        self.assertEqual(
            trainer_module._training_contract_sha256(first),
            trainer_module._training_contract_sha256(second),
        )

    def test_stage_parent_is_an_immutable_resume_contract_field(self):
        first = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run-a",
            train_sources=[{"manifest": "data.jsonl"}],
            init_adapter_from="checkpoint-a",
            stage_parent_checkpoint="checkpoint-a",
        )
        second = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run-b",
            train_sources=[{"manifest": "data.jsonl"}],
            resume_from="resume-b",
            stage_parent_checkpoint="checkpoint-b",
        )
        self.assertNotEqual(
            trainer_module._training_contract_sha256(first),
            trainer_module._training_contract_sha256(second),
        )

    def test_stage_parent_must_match_fresh_adapter_initializer(self):
        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            init_adapter_from="checkpoint-a",
            stage_parent_checkpoint="checkpoint-b",
        )
        with self.assertRaisesRegex(ValueError, "must match stage_parent_checkpoint"):
            config.validate()

    def test_exported_core_can_define_a_fresh_lora_stage_parent(self):
        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            core_model_dir="exported-core",
            stage_parent_checkpoint="full-checkpoint",
            mode="lora",
        )
        config.validate()
        self.assertIsNone(config.init_adapter_from)

    def test_exported_core_can_define_a_fresh_full_stage_parent(self):
        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            core_model_dir="exported-core",
            stage_parent_checkpoint="full-checkpoint",
            mode="full",
            strategy="fsdp",
        )
        config.validate()



    def test_scheduler_horizon_remains_full_during_gate(self):
        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=2e-6)
        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            max_steps=53640,
            scheduler_steps=53640,
            warmup_ratio=0.03,
            learning_rate=2e-6,
        )
        scheduler = trainer_module._build_scheduler(optimizer, config)
        for _ in range(200):
            optimizer.step()
            scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[0], 0.0)
        self.assertLess(scheduler.get_last_lr()[0], config.learning_rate)

    def test_checkpoint_precision_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint-00001000"
            checkpoint.mkdir()
            source = root / "PRECISION.json"
            first = {
                "status": "passed",
                "verified_at_step": 1,
                "trainable_master_dtype": "float32",
                "optimizer_state_dtype": "float32",
            }
            source.write_text(
                json.dumps(first, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            report, snapshot, digest = trainer_module._snapshot_precision_report(
                root,
                checkpoint,
            )
            source.write_text(
                json.dumps({**first, "verified_at_step": 1000}, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            self.assertEqual(report, first)
            self.assertEqual(snapshot, checkpoint / "PRECISION.json")
            self.assertEqual(json.loads(snapshot.read_text(encoding="utf-8")), first)
            self.assertEqual(trainer_module.hashlib.sha256(snapshot.read_bytes()).hexdigest(), digest)

    def test_cache_only_features_fail_closed_on_miss(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audio = root / "sample.wav"
            sf.write(audio, torch.zeros(2400).numpy(), 24000)
            record = SpeechRecord(
                record_id="sample",
                audio_path=str(audio),
                text="text",
                speaker_id="speaker",
            )
            features = FrozenAudioFeatures(
                None,
                None,
                device=torch.device("cpu"),
                patch_size=4,
                cache_dir=root / "cache",
                cache_metadata={
                    "sample_rate": 24000,
                    "downsample_rate": 960,
                    "hidden_size": 64,
                },
            )
            with self.assertRaisesRegex(FileNotFoundError, "cache-only mode"):
                features.get(record)
            cache_path = features._cache_path(record)
            assert cache_path is not None
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "latents": torch.ones(1, 4, 64, dtype=torch.float16),
                    "speaker": torch.ones(1, 512),
                    "sample_rate": 24000,
                    "audio_path": str(audio),
                },
                cache_path,
            )
            latents, speaker = features.get(record)
            self.assertEqual(tuple(latents.shape), (1, 4, 64))
            self.assertEqual(tuple(speaker.shape), (1, 512))


class AlignmentTest(unittest.TestCase):
    def test_condition_window_never_contains_current_target(self):
        # Prompt states 10,11; target states 20,21,22.
        hidden = torch.tensor([[[10.0], [11.0], [20.0], [21.0], [22.0]]])
        windows = build_condition_windows(
            hidden,
            prompt_patches=2,
            target_patch_indices=torch.tensor([0, 1, 2]),
            history_patches=1,
        )
        self.assertEqual(windows.squeeze(-1).tolist(), [[10.0, 11.0], [11.0, 20.0], [20.0, 21.0]])

    def test_latent_history_and_current_alignment(self):
        prompt = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
        target = torch.tensor([[[5.0], [6.0], [7.0], [8.0]]])
        histories, current = build_latent_history_windows(
            prompt,
            target,
            target_patch_indices=torch.tensor([0, 1]),
            patch_size=2,
            history_length=2,
        )
        self.assertEqual(histories.squeeze(-1).tolist(), [[3.0, 4.0], [5.0, 6.0]])
        self.assertEqual(current.squeeze(-1).tolist(), [[5.0, 6.0], [7.0, 8.0]])

    def test_auto_stop_weight_balances_one_end_against_all_continuations(self):
        logits = torch.zeros(11, requires_grad=True)
        loss, positive, max_negative, margin, effective = balanced_stop_loss(
            logits,
            positive_weight="auto",
            positive_weight_max=64.0,
        )
        self.assertEqual(effective, 10.0)
        self.assertAlmostEqual(float(positive.detach()), 0.5)
        self.assertAlmostEqual(float(max_negative.detach()), 0.5)
        self.assertAlmostEqual(float(margin.detach()), 0.0)
        # Ten negatives and one positive weighted by ten contribute equally.
        self.assertAlmostEqual(
            float(loss.detach()), 20.0 * 0.69314718056 / 11.0, places=5
        )
        loss.backward()
        self.assertLess(float(logits.grad[-1]), 0.0)

    def test_auto_stop_weight_is_capped(self):
        logits = torch.zeros(101)
        *_, effective = balanced_stop_loss(
            logits,
            positive_weight="auto",
            positive_weight_max=32.0,
        )
        self.assertEqual(effective, 32.0)


class LoRATest(unittest.TestCase):
    def test_zero_initialized_adapter_preserves_output(self):
        model = nn.Sequential(nn.Linear(4, 5), nn.ReLU(), nn.Linear(5, 3))
        inputs = torch.randn(2, 4)
        expected = model(inputs).detach()
        config = LoRAConfig(rank=2, alpha=4, dropout=0, target_patterns=(r"^0$", r"^2$"))
        replaced = inject_lora(model, config)
        actual = model(inputs).detach()
        self.assertEqual(replaced, ["0", "2"])
        self.assertTrue(torch.equal(expected, actual))
        trainable, total = trainable_parameter_counts(model)
        self.assertGreater(trainable, 0)
        self.assertLess(trainable, total)

    def test_route_mask_preserves_prefix_and_updates_target(self):
        model = nn.Sequential(nn.Linear(2, 2, bias=False))
        with torch.no_grad():
            model[0].weight.copy_(torch.eye(2))
        config = LoRAConfig(
            rank=1,
            alpha=1,
            dropout=0,
            target_patterns=(r"^0$",),
            routing="target_audio",
        )
        inject_lora(model, config)
        with torch.no_grad():
            model[0].lora_A.fill_(1.0)
            model[0].lora_B.fill_(1.0)
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        base = model[0].base(inputs)
        set_lora_route_mask(model, torch.tensor([[[0.0], [1.0]]]))
        routed = model(inputs)
        self.assertTrue(torch.equal(routed[:, :1], base[:, :1]))
        self.assertFalse(torch.equal(routed[:, 1:], base[:, 1:]))

    def test_routing_round_trip_defaults_old_adapters_to_global(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "adapter_config.json"
            LoRAConfig(routing="target_audio").save_json(path)
            self.assertEqual(LoRAConfig.from_json(path).routing, "target_audio")
            value = json.loads(path.read_text(encoding="utf-8"))
            value.pop("routing")
            path.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(LoRAConfig.from_json(path).routing, "global")

    def test_semantic_decode_route_enables_text_anchor_and_target_only(self):
        mask = build_backbone_route_mask(
            "semantic_decode",
            text_tokens=3,
            prompt_patches=2,
            target_patches=2,
            reference=torch.zeros(1),
        )
        self.assertIsNotNone(mask)
        self.assertEqual(mask.flatten().tolist(), [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])

    def test_legacy_target_audio_route_semantics_are_unchanged(self):
        mask = build_backbone_route_mask(
            "target_audio",
            text_tokens=2,
            prompt_patches=2,
            target_patches=2,
            reference=torch.zeros(1),
        )
        self.assertIsNotNone(mask)
        self.assertEqual(mask.flatten().tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0])

    def test_extra_trainable_bf16_parameter_gets_fp32_master_copy(self):
        model = nn.Sequential(nn.Linear(2, 2).to(torch.bfloat16))
        inject_lora(
            model,
            LoRAConfig(
                rank=1,
                alpha=1,
                dropout=0,
                target_patterns=(r"^0$",),
                train_extra_patterns=(r"^0\.base\.bias$",),
            ),
        )
        self.assertEqual(model[0].base.bias.dtype, torch.float32)
        self.assertTrue(model[0].base.bias.requires_grad)

    def test_resume_and_stage_initialization_are_mutually_exclusive(self):
        config = TrainerConfig(
            pretrained_model_dir="model",
            output_dir="run",
            train_sources=[{"manifest": "data.jsonl"}],
            resume_from="resume",
            init_adapter_from="adapter",
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
