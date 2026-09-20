"""Distributed training entry point for FireRedTTS3-Base."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from fireredtts3.campp.campp import CamppEmbedding
from fireredtts3.llm.dit import DiTBlock as FlowDiTBlock
from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCore
from fireredtts3.llm.modules import DiTBlock as PatchDiTBlock
from fireredtts3.redae.redae import RedAE
from fireredtts3.utils.text_tokenizer import (
    MULTI_DIALECT_TAGS,
    MULTI_LANG_TAGS,
    load_text_tokenizer,
)


from .data import (
    DistributedWeightedSampler,
    PromptPairedDataset,
    SourceSpec,
    dataset_summary,
    load_mixture_records,
    split_records_by_group,
)
from .features import FrozenAudioFeatures
from .lora import (
    DEFAULT_BACKBONE_PATTERNS,
    DEFAULT_DIT_PATTERNS,
    DEFAULT_PATCH_ENCODER_PATTERNS,
    LoRAConfig,
    inject_lora,
    load_lora_adapter,
    save_lora_adapter,
    trainable_parameter_counts,
)
from .model import FireRedTTS3ForTraining


@dataclass
class TrainerConfig:
    pretrained_model_dir: str
    output_dir: str
    train_sources: list[dict[str, Any]]
    core_model_dir: str | None = None
    validation_sources: list[dict[str, Any]] = field(default_factory=list)
    validation_fraction: float = 0.01
    split_group_field: str = "video_id"
    language_ids: list[str] = field(default_factory=lambda: ["my"])
    language_tag: str = "Burmese"
    text_separator: str = " "
    # ``legacy`` matches the released model: one target-language tag before
    # <|sot|>.  ``prompt_target_tags`` makes a cross-lingual prompt explicit by
    # tagging the prompt language first and the target language at the boundary.
    text_sequence_format: str = "legacy"
    prompt_language_tags: dict[str, str] = field(default_factory=dict)

    mode: str = "lora"  # lora | full
    strategy: str = "auto"  # auto | ddp | fsdp | single
    lora: dict[str, Any] = field(default_factory=dict)
    gradient_checkpointing: bool = True
    # Forward/backward compute dtype.  Trainable/master parameters are always
    # FP32; FSDP casts them to this dtype only while computing.
    dtype: str = "bfloat16"

    seed: int = 42
    max_steps: int = 9656
    scheduler_steps: int | None = None
    samples_per_epoch: int | None = None
    expected_train_records: int | None = None
    expected_world_size: int | None = None
    expected_steps_per_epoch: int | None = None
    gradient_accumulation_steps: int = 1
    # This isolated version deliberately starts a new checkpoint contract.
    data_order_resume_version: str = "epoch_seeded_v2"
    learning_rate: float = 1.0e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1.0e-8
    # False uses single-tensor AdamW to avoid a model-sized foreach temporary.
    # None retains the legacy optimizer dispatch and resume-contract behavior.
    optimizer_foreach: bool | None = None
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0

    min_target_seconds: float = 0.4
    max_target_seconds: float = 20.0
    min_prompt_seconds: float = 1.0
    max_prompt_seconds: float = 10.0
    max_text_tokens: int = 768
    max_flow_patches: int = 24
    cfg_dropout_prob: float = 0.1
    stop_loss_weight: float = 0.1
    stop_positive_weight: float | str = 1.0
    stop_positive_weight_max: float = 64.0
    feature_cache_dir: str | None = None
    feature_cache_only: bool = False
    check_audio_exists: bool = True
    fail_on_nonfinite: bool = True

    num_workers: int = 0
    log_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 500
    save_at_steps: list[int] = field(default_factory=list)
    max_eval_samples: int = 256
    keep_last_checkpoints: int = 3
    resume_from: str | None = None
    init_adapter_from: str | None = None
    stage_parent_checkpoint: str | None = None

    # A disabled-by-default, deterministic early gate.  New full runs enable
    # this at step 200; legacy experiment contracts remain unchanged.
    health_check_step: int = 0
    health_loss_check_step: int = 0
    health_parameter_samples_per_tensor: int = 32
    health_min_changed_fraction: float = 0.01
    health_min_relative_rms_delta: float = 1.0e-7
    health_min_probe_flow_loss_relative_decrease: float = 0.0
    health_require_stop_head_change: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainerConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {item.name for item in cls.__dataclass_fields__.values()}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown config keys: {unknown}")
        return cls(**value)

    def validate(self) -> None:
        if self.mode not in {"lora", "full"}:
            raise ValueError("mode must be 'lora' or 'full'")
        if self.strategy not in {"auto", "ddp", "fsdp", "single"}:
            raise ValueError("strategy must be auto/ddp/fsdp/single")
        if self.text_sequence_format not in {"legacy", "prompt_target_tags"}:
            raise ValueError("text_sequence_format must be legacy/prompt_target_tags")
        supported_tags = set(MULTI_LANG_TAGS + MULTI_DIALECT_TAGS)
        target_tag = f"<|{self.language_tag}|>"
        if target_tag not in supported_tags:
            raise ValueError(f"unsupported language tag: {target_tag}")
        unsupported_prompt_tags = sorted(
            f"{language_id}=<|{tag}|>"
            for language_id, tag in self.prompt_language_tags.items()
            if f"<|{tag}|>" not in supported_tags
        )
        if unsupported_prompt_tags:
            raise ValueError(
                "unsupported prompt language tags: " + ", ".join(unsupported_prompt_tags)
            )
        if self.text_sequence_format == "prompt_target_tags":
            missing_prompt_tags = sorted(set(self.language_ids) - set(self.prompt_language_tags))
            if missing_prompt_tags:
                raise ValueError(
                    "prompt_target_tags requires prompt_language_tags for configured "
                    f"language_ids: {missing_prompt_tags}"
                )
        if not self.train_sources:
            raise ValueError("train_sources is required")
        if self.data_order_resume_version != "epoch_seeded_v2":
            raise ValueError("this trainer requires data_order_resume_version=epoch_seeded_v2")
        if self.num_workers != 0:
            raise ValueError("epoch_seeded_v2 exact resume currently requires num_workers=0")
        if self.max_steps <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("max_steps and gradient_accumulation_steps must be positive")
        if self.scheduler_steps is not None and self.scheduler_steps <= 0:
            raise ValueError("scheduler_steps must be positive when provided")
        for name in ("expected_train_records", "expected_world_size", "expected_steps_per_epoch"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when provided")
        if len(self.save_at_steps) != len(set(self.save_at_steps)):
            raise ValueError("save_at_steps must not contain duplicates")
        if any(step <= 0 or step > self.max_steps for step in self.save_at_steps):
            raise ValueError("save_at_steps entries must be in [1, max_steps]")
        if self.feature_cache_only and not self.feature_cache_dir:
            raise ValueError("feature_cache_only requires feature_cache_dir")
        if self.optimizer_foreach is not None and type(self.optimizer_foreach) is not bool:
            raise ValueError("optimizer_foreach must be a boolean or None")
        if self.dtype != "bfloat16":
            raise ValueError("the released checkpoints currently support dtype='bfloat16' only")
        if isinstance(self.stop_positive_weight, str):
            if self.stop_positive_weight != "auto":
                raise ValueError("stop_positive_weight must be a positive number or 'auto'")
        elif not math.isfinite(float(self.stop_positive_weight)) or float(
            self.stop_positive_weight
        ) <= 0:
            raise ValueError("stop_positive_weight must be finite and positive")
        if not math.isfinite(self.stop_positive_weight_max) or self.stop_positive_weight_max <= 0:
            raise ValueError("stop_positive_weight_max must be finite and positive")
        if not 0.0 <= self.cfg_dropout_prob < 1.0:
            raise ValueError("cfg_dropout_prob must be in [0, 1)")
        if self.health_check_step < 0 or self.health_check_step > self.max_steps:
            raise ValueError("health_check_step must be in [0, max_steps]")
        if self.health_loss_check_step < 0 or self.health_loss_check_step > self.max_steps:
            raise ValueError("health_loss_check_step must be in [0, max_steps]")
        if (
            self.health_check_step
            and self.health_loss_check_step
            and self.health_loss_check_step < self.health_check_step
        ):
            raise ValueError("health_loss_check_step must not precede health_check_step")
        if self.health_check_step and self.health_parameter_samples_per_tensor <= 0:
            raise ValueError("health_parameter_samples_per_tensor must be positive")
        if not 0.0 <= self.health_min_changed_fraction <= 1.0:
            raise ValueError("health_min_changed_fraction must be in [0, 1]")
        if self.health_min_relative_rms_delta < 0:
            raise ValueError("health_min_relative_rms_delta must be non-negative")
        if self.health_min_probe_flow_loss_relative_decrease < 0:
            raise ValueError(
                "health_min_probe_flow_loss_relative_decrease must be non-negative"
            )
        if self.mode == "full" and self.strategy == "ddp":
            raise ValueError("full training on 24GB GPUs requires FSDP, not replicated DDP")
        if self.resume_from and self.init_adapter_from:
            raise ValueError("resume_from and init_adapter_from are mutually exclusive")
        if self.init_adapter_from and self.mode != "lora":
            raise ValueError("init_adapter_from currently supports mode='lora' only")
        if self.core_model_dir and self.mode == "full" and not self.stage_parent_checkpoint:
            raise ValueError(
                "full training from core_model_dir requires stage_parent_checkpoint provenance"
            )
        if (
            self.init_adapter_from
            and self.stage_parent_checkpoint
            and Path(self.init_adapter_from) != Path(self.stage_parent_checkpoint)
        ):
            raise ValueError("init_adapter_from must match stage_parent_checkpoint")
        if self.stage_parent_checkpoint and not (
            self.init_adapter_from or self.core_model_dir or self.resume_from
        ):
            raise ValueError(
                "stage_parent_checkpoint requires init_adapter_from or core_model_dir "
                "on a fresh stage"
            )


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed(strategy: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1 or strategy == "fsdp"
    if distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("FireRedTTS3 training requires CUDA")
    torch.cuda.set_device(local_rank)
    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=dist.is_initialized(),
        device=torch.device("cuda", local_rank),
    )


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def log(message: str, context: DistributedContext, *, all_ranks: bool = False) -> None:
    if context.is_main or all_ranks:
        prefix = f"[rank {context.rank}] " if all_ranks else ""
        print(prefix + message, flush=True)


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def _source_specs(values: Sequence[Mapping[str, Any]]) -> list[SourceSpec]:
    return [SourceSpec.from_mapping(value) for value in values]


def _build_datasets(config: TrainerConfig) -> tuple[PromptPairedDataset, PromptPairedDataset | None]:
    common = dict(
        language_ids=set(config.language_ids),
        min_duration=config.min_target_seconds,
        max_duration=config.max_target_seconds,
        check_audio_exists=config.check_audio_exists,
    )
    train_records = load_mixture_records(_source_specs(config.train_sources), **common)
    valid_records = []
    if config.validation_sources:
        valid_records = load_mixture_records(_source_specs(config.validation_sources), **common)
    elif config.validation_fraction > 0:
        train_records, valid_records = split_records_by_group(
            train_records,
            config.validation_fraction,
            seed=config.seed,
            group_field=config.split_group_field,
        )

    paired_kwargs = dict(
        seed=config.seed,
        prompt_min_duration=config.min_prompt_seconds,
        prompt_max_duration=config.max_prompt_seconds,
    )
    train_dataset = PromptPairedDataset(train_records, **paired_kwargs)
    valid_dataset = PromptPairedDataset(valid_records, **paired_kwargs) if valid_records else None
    if (
        config.expected_train_records is not None
        and len(train_dataset) != config.expected_train_records
    ):
        raise RuntimeError(
            "training record count changed: "
            f"expected={config.expected_train_records}, actual={len(train_dataset)}"
        )
    return train_dataset, valid_dataset


def _single_item_collate(batch: list[Any]) -> Any:
    if len(batch) != 1:
        raise ValueError("FireRedTTS3 PatchEncoder supports per-device batch size 1")
    return batch[0]


def _build_loaders(
    config: TrainerConfig,
    context: DistributedContext,
    train_dataset: PromptPairedDataset,
    valid_dataset: PromptPairedDataset | None,
) -> tuple[DataLoader, DataLoader | None, Any, Any]:
    source_count = len({item.source_name for item in train_dataset.records})
    if source_count > 1 or config.samples_per_epoch is not None:
        train_sampler = DistributedWeightedSampler(
            [item.sampling_mass for item in train_dataset.records],
            num_replicas=context.world_size,
            rank=context.rank,
            samples_per_epoch=config.samples_per_epoch,
            seed=config.seed,
        )
    elif context.distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )
    else:
        # Use the same epoch+seed contract as the distributed branch. An
        # unseeded RandomSampler would create a different permutation after
        # restoring the checkpoint's current global RNG and skipping batches.
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=1,
            rank=0,
            shuffle=True,
            seed=config.seed,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=False,
        collate_fn=_single_item_collate,
        persistent_workers=config.num_workers > 0,
        # DataLoader allocates a base seed even with zero workers. Isolate that
        # draw so iterator reconstruction never advances model/global CPU RNG.
        generator=torch.Generator().manual_seed(config.seed + context.rank),
    )

    valid_sampler = None
    valid_loader = None
    if valid_dataset is not None:
        valid_sampler = (
            DistributedSampler(
                valid_dataset,
                num_replicas=context.world_size,
                rank=context.rank,
                shuffle=False,
                drop_last=False,
            )
            if context.distributed
            else torch.utils.data.SequentialSampler(valid_dataset)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=1,
            sampler=valid_sampler,
            num_workers=config.num_workers,
            pin_memory=False,
            collate_fn=_single_item_collate,
            persistent_workers=config.num_workers > 0,
            generator=torch.Generator().manual_seed(config.seed + context.rank + 1),
        )
    return train_loader, valid_loader, train_sampler, valid_sampler


def _lora_config(value: Mapping[str, Any]) -> LoRAConfig:
    scopes = value.get("scopes", ["backbone", "dit"])
    scope_patterns: dict[str, Sequence[str]] = {
        "backbone": DEFAULT_BACKBONE_PATTERNS,
        "dit": DEFAULT_DIT_PATTERNS,
        "patch_encoder": DEFAULT_PATCH_ENCODER_PATTERNS,
    }
    if value.get("target_patterns"):
        patterns = tuple(value["target_patterns"])
    else:
        unknown = sorted(set(scopes) - set(scope_patterns))
        if unknown:
            raise ValueError(f"unknown LoRA scopes: {unknown}")
        patterns = tuple(pattern for scope in scopes for pattern in scope_patterns[scope])
    return LoRAConfig(
        rank=int(value.get("rank", 8)),
        alpha=float(value.get("alpha", 16.0)),
        dropout=float(value.get("dropout", 0.05)),
        target_patterns=patterns,
        train_extra_patterns=tuple(value.get("train_extra_patterns", ())),
        routing=str(value.get("routing", "global")),
    )


def _enable_gradient_checkpointing(core: FireRedTTS3BaseCore, enabled: bool) -> None:
    core.patch_encoder.gradient_checkpointing = bool(enabled)
    core.dit.gradient_checkpointing = bool(enabled)
    core.backbone_llm.config.use_cache = False
    if enabled:
        try:
            core.backbone_llm.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            core.backbone_llm.gradient_checkpointing_enable()


def _load_core(
    config: TrainerConfig,
    context: DistributedContext,
) -> tuple[FireRedTTS3ForTraining, LoRAConfig | None]:
    model_path = (
        Path(config.core_model_dir)
        if config.core_model_dir
        else Path(config.pretrained_model_dir) / "fireredtts3_base"
    )
    if config.core_model_dir:
        required = [
            model_path / "config.json",
            model_path / "model.safetensors",
            model_path / "export_metadata.json",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"exported core is incomplete; missing={missing}")
        export = json.loads(
            (model_path / "export_metadata.json").read_text(encoding="utf-8")
        )
        if (
            export.get("status") != "passed"
            or export.get("checkpoint_type") != "full_fsdp_export"
        ):
            raise RuntimeError(f"invalid exported core provenance: {export}")
        if config.stage_parent_checkpoint and Path(
            export.get("source_checkpoint", "")
        ).resolve() != Path(config.stage_parent_checkpoint).resolve():
            raise RuntimeError(
                "exported core source does not match stage_parent_checkpoint: "
                f"export={export.get('source_checkpoint')}, "
                f"parent={config.stage_parent_checkpoint}"
            )
        if int(export.get("weights", {}).get("bytes", -1)) != (
            model_path / "model.safetensors"
        ).stat().st_size:
            raise RuntimeError("exported core weight size does not match provenance")
        if export.get("config", {}).get("sha256") != hashlib.sha256(
            (model_path / "config.json").read_bytes()
        ).hexdigest():
            raise RuntimeError("exported core config hash does not match provenance")
    # FSDP mixed precision only preserves full-precision optimizer weights when
    # the incoming parameters are full precision.  Loading a full-training core
    # as BF16 first permanently discards that master copy and makes AdamW state
    # BF16 as well, causing small updates to quantize away.
    storage_dtype = torch.float32 if config.mode == "full" else torch.bfloat16
    core = FireRedTTS3BaseCore.from_pretrained(
        str(model_path),
        torch_dtype=storage_dtype,
    )
    log(f"loaded core storage dtype={storage_dtype} mode={config.mode}", context)
    core.backbone_llm.config.use_cache = False
    adapter_config = None
    if config.mode == "lora":
        adapter_source = config.resume_from or config.init_adapter_from
        if adapter_source:
            if not (Path(adapter_source) / "adapter_config.json").is_file():
                raise FileNotFoundError(
                    f"adapter_config.json not found in adapter source: {adapter_source}"
                )
            adapter_config = load_lora_adapter(core, adapter_source)
        else:
            adapter_config = _lora_config(config.lora)
            replaced = inject_lora(core, adapter_config)
            log(f"injected LoRA into {len(replaced)} Linear modules", context)
    else:
        for parameter in core.parameters():
            parameter.requires_grad_(True)

    _enable_gradient_checkpointing(core, config.gradient_checkpointing)
    training_model = FireRedTTS3ForTraining(core)
    trainable, total = trainable_parameter_counts(training_model)
    log(
        f"parameters trainable={trainable:,} total={total:,} ({100.0 * trainable / total:.4f}%)",
        context,
    )
    return training_model, adapter_config


def _wrap_model(
    model: FireRedTTS3ForTraining,
    config: TrainerConfig,
    context: DistributedContext,
) -> tuple[torch.nn.Module, str]:
    strategy = config.strategy
    if strategy == "auto":
        strategy = "fsdp" if config.mode == "full" else ("ddp" if context.distributed else "single")
    if strategy == "single":
        model.to(context.device)
        return model, strategy
    if strategy == "ddp":
        if not context.distributed:
            raise ValueError("strategy=ddp requires torchrun with WORLD_SIZE > 1")
        model.to(context.device)
        wrapped = DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        return wrapped, strategy
    if strategy == "fsdp":
        if not context.distributed:
            raise ValueError("strategy=fsdp requires torchrun (one process is allowed)")
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

        auto_wrap = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={Qwen3DecoderLayer, FlowDiTBlock, PatchDiTBlock},
        )
        model.to(context.device)
        wrapped = FSDP(
            model,
            auto_wrap_policy=auto_wrap,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=False,
                cast_forward_inputs=True,
            ),
            device_id=context.device,
            sync_module_states=True,
            use_orig_params=True,
            limit_all_gathers=True,
        )
        return wrapped, strategy
    raise AssertionError(strategy)


def _unwrap_training_model(model: torch.nn.Module) -> FireRedTTS3ForTraining:
    if isinstance(model, DistributedDataParallel):
        return model.module
    # FSDP forwards attribute access to the wrapped module in current PyTorch,
    # but _fsdp_wrapped_module is explicit and stable for checkpoint helpers.
    return getattr(model, "_fsdp_wrapped_module", model)


def _dtype_label(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _assert_trainable_parameters_fp32(
    model: torch.nn.Module,
    context: DistributedContext,
) -> dict[str, int]:
    local_fp32 = 0
    local_other = 0
    local_bad: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or not parameter.is_floating_point():
            continue
        if parameter.dtype == torch.float32:
            local_fp32 += parameter.numel()
        else:
            local_other += parameter.numel()
            if len(local_bad) < 8:
                local_bad.append(f"{name}:{parameter.dtype}:{parameter.numel()}")
    counts = torch.tensor(
        [local_fp32, local_other], device=context.device, dtype=torch.int64
    )
    if context.distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    report = {
        "float32_values": int(counts[0].item()),
        "non_float32_values": int(counts[1].item()),
    }
    if report["non_float32_values"]:
        if local_bad:
            log(f"non-FP32 trainable parameters: {local_bad}", context, all_ranks=True)
        raise RuntimeError(
            "trainable/master parameters must be FP32 before optimizer creation; "
            f"summary={report}"
        )
    return report


def _optimizer_precision_report(
    optimizer: torch.optim.Optimizer,
    context: DistributedContext,
) -> dict[str, int]:
    local_fp32 = 0
    local_other = 0
    local_tensors = 0
    local_bad: list[str] = []
    for state in optimizer.state.values():
        for name, value in state.items():
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            local_tensors += 1
            if value.dtype == torch.float32:
                local_fp32 += value.numel()
            else:
                local_other += value.numel()
                if len(local_bad) < 8:
                    local_bad.append(f"{name}:{value.dtype}:{value.numel()}")
    counts = torch.tensor(
        [local_fp32, local_other, local_tensors],
        device=context.device,
        dtype=torch.int64,
    )
    if context.distributed:
        dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    report = {
        "float32_values": int(counts[0].item()),
        "non_float32_values": int(counts[1].item()),
        "floating_state_tensors": int(counts[2].item()),
    }
    if report["floating_state_tensors"] == 0:
        raise RuntimeError("optimizer has no initialized floating-point state")
    if report["non_float32_values"]:
        if local_bad:
            log(f"non-FP32 optimizer state: {local_bad}", context, all_ranks=True)
        raise RuntimeError(f"AdamW floating-point state must be FP32; summary={report}")
    return report


def _precision_report_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "PRECISION.json"


def _write_precision_report(
    config: TrainerConfig,
    parameter_precision: Mapping[str, int],
    optimizer_precision: Mapping[str, int],
    context: DistributedContext,
    *,
    verified_at_step: int,
) -> None:
    if context.is_main:
        _atomic_json(
            _precision_report_path(config.output_dir),
            {
                "schema_version": 1,
                "status": "passed",
                "verified_at_step": int(verified_at_step),
                "compute_dtype": config.dtype,
                "trainable_master_dtype": "float32",
                "optimizer_state_dtype": "float32",
                "parameters": dict(parameter_precision),
                "optimizer": dict(optimizer_precision),
            },
        )
    barrier()


def _parameter_group(name: str) -> str:
    if "stop_head." in name:
        return "stop_head"
    if "backbone_llm." in name:
        return "backbone"
    if ".dit." in name or name.startswith("dit."):
        return "dit"
    if "patch_encoder." in name:
        return "patch_encoder"
    return "other"


@dataclass
class _ParameterProbe:
    name: str
    group: str
    parameter: torch.nn.Parameter
    indices: torch.Tensor
    initial: torch.Tensor


class ParameterChangeMonitor:
    """Sample sharded trainable parameters without retaining model-sized copies."""

    GROUPS = ("all", "stop_head", "backbone", "dit", "patch_encoder", "other")

    def __init__(self, model: torch.nn.Module, samples_per_tensor: int):
        self.probes: list[_ParameterProbe] = []
        for name, parameter in model.named_parameters():
            if (
                not parameter.requires_grad
                or not parameter.is_floating_point()
                or parameter.numel() == 0
            ):
                continue
            count = min(int(samples_per_tensor), parameter.numel())
            if count == parameter.numel():
                indices = torch.arange(count, dtype=torch.long)
            else:
                indices = torch.linspace(
                    0, parameter.numel() - 1, steps=count, dtype=torch.float64
                ).round().to(torch.long).unique()
            current = parameter.detach().reshape(-1).index_select(
                0, indices.to(parameter.device)
            )
            initial = current.float().cpu().clone()
            if not torch.isfinite(initial).all():
                raise RuntimeError(f"non-finite initial parameter sample: {name}")
            self.probes.append(
                _ParameterProbe(
                    name=name,
                    group=_parameter_group(name),
                    parameter=parameter,
                    indices=indices,
                    initial=initial,
                )
            )
        if not self.probes:
            raise RuntimeError("parameter-change monitor found no local trainable samples")

    @torch.no_grad()
    def report(self, context: DistributedContext) -> dict[str, dict[str, float | int]]:
        # Columns: sampled, exactly changed, squared delta, squared baseline.
        sums = torch.zeros(
            len(self.GROUPS), 4, device=context.device, dtype=torch.float64
        )
        maxima = torch.zeros(len(self.GROUPS), device=context.device, dtype=torch.float64)
        group_index = {name: index for index, name in enumerate(self.GROUPS)}
        for probe in self.probes:
            current = probe.parameter.detach().reshape(-1).index_select(
                0, probe.indices.to(probe.parameter.device)
            ).float().cpu()
            if current.shape != probe.initial.shape:
                raise RuntimeError(f"parameter shard layout changed for probe {probe.name}")
            delta = current - probe.initial
            for group in ("all", probe.group):
                index = group_index[group]
                sums[index, 0] += delta.numel()
                sums[index, 1] += (delta != 0).sum().item()
                sums[index, 2] += delta.double().square().sum().item()
                sums[index, 3] += probe.initial.double().square().sum().item()
                if delta.numel():
                    maxima[index] = torch.maximum(
                        maxima[index],
                        delta.abs().max().to(device=context.device, dtype=torch.float64),
                    )
        if context.distributed:
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        report: dict[str, dict[str, float | int]] = {}
        for index, group in enumerate(self.GROUPS):
            sampled = int(sums[index, 0].item())
            changed = int(sums[index, 1].item())
            delta_sq = float(sums[index, 2].item())
            base_sq = float(sums[index, 3].item())
            report[group] = {
                "sampled_values": sampled,
                "changed_values": changed,
                "changed_fraction": (changed / sampled if sampled else 0.0),
                "rms_delta": (math.sqrt(delta_sq / sampled) if sampled else 0.0),
                "relative_rms_delta": (
                    math.sqrt(delta_sq / base_sq)
                    if base_sq > 0
                    else (math.sqrt(delta_sq / sampled) if sampled else 0.0)
                ),
                "max_abs_delta": float(maxima[index].item()),
            }
        return report


def _build_optimizer(model: torch.nn.Module, config: TrainerConfig) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=config.learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        eps=config.adam_eps,
        weight_decay=config.weight_decay,
        foreach=config.optimizer_foreach,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    scheduler_steps = config.scheduler_steps or config.max_steps
    warmup_steps = int(round(scheduler_steps * config.warmup_ratio))

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(scheduler_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _load_frozen_features(
    config: TrainerConfig,
    context: DistributedContext,
    patch_size: int,
) -> FrozenAudioFeatures:
    root = Path(config.pretrained_model_dir)
    if config.feature_cache_only:
        redae_config = json.loads((root / "redae" / "config.json").read_text(encoding="utf-8"))
        return FrozenAudioFeatures(
            None,
            None,
            device=context.device,
            patch_size=patch_size,
            cache_dir=config.feature_cache_dir,
            cache_metadata={
                "sample_rate": int(redae_config["audio_sample_rate"]),
                "downsample_rate": int(redae_config["audio_patch_size"])
                * int(redae_config["enc_extra_downsample_rate"]),
                "hidden_size": int(redae_config["bottleneck_dim"]),
            },
        )
    redae = RedAE.from_pretrained(str(root / "redae"), torch_dtype=torch.bfloat16).to(context.device)
    speaker = CamppEmbedding(str(root / "campp" / "campplus_voxceleb.bin")).to(context.device)
    return FrozenAudioFeatures(
        redae,
        speaker,
        device=context.device,
        patch_size=patch_size,
        cache_dir=config.feature_cache_dir,
    )


def _build_text_tokens(
    tokenizer: Any,
    config: TrainerConfig,
    prompt_text: str,
    target_text: str,
    device: torch.device,
    *,
    prompt_language_id: str | None = None,
) -> torch.Tensor:
    target_lang_tag = f"<|{config.language_tag}|>"
    if target_lang_tag not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
        raise ValueError(f"unsupported language tag: {target_lang_tag}")
    sequence_lang_tag = target_lang_tag
    target_boundary = ""
    if config.text_sequence_format == "prompt_target_tags":
        prompt_language_tag = config.prompt_language_tags.get(prompt_language_id or "")
        if prompt_language_tag is None:
            raise ValueError(
                "prompt_target_tags has no prompt language tag for "
                f"language_id={prompt_language_id!r}"
            )
        sequence_lang_tag = f"<|{prompt_language_tag}|>"
        if sequence_lang_tag not in MULTI_LANG_TAGS + MULTI_DIALECT_TAGS:
            raise ValueError(f"unsupported prompt language tag: {sequence_lang_tag}")
        target_boundary = target_lang_tag
    separator = config.text_separator if prompt_text and target_text else ""
    content = (
        f"{sequence_lang_tag}<|sot|>{prompt_text}{separator}"
        f"{target_boundary}{target_text}<|eot|>"
    )
    token_ids = tokenizer(content, add_special_tokens=False, truncation=False)["input_ids"]
    if len(token_ids) > config.max_text_tokens:
        raise ValueError(
            f"text has {len(token_ids)} tokens, above max_text_tokens={config.max_text_tokens}; "
            "do not truncate text without matching audio"
        )
    return torch.tensor([token_ids], dtype=torch.long, device=device)


def _batch_to_inputs(
    batch: Mapping[str, Any],
    tokenizer: Any,
    features: FrozenAudioFeatures,
    config: TrainerConfig,
    context: DistributedContext,
) -> dict[str, Any]:
    target = batch["target"]
    prompt = batch["prompt"]
    prompt_latents, prompt_speaker = features.get(prompt)
    target_latents, _ = features.get(target)
    text_tokens = _build_text_tokens(
        tokenizer,
        config,
        prompt.text,
        target.text,
        context.device,
        prompt_language_id=prompt.language_id,
    )
    return {
        "spk_emb": prompt_speaker,
        "text_tokens": text_tokens,
        "prompt_latents": prompt_latents,
        "target_latents": target_latents,
        "sample_weight": target.loss_weight,
        "max_flow_patches": config.max_flow_patches,
        "cfg_dropout_prob": config.cfg_dropout_prob,
        "stop_loss_weight": config.stop_loss_weight,
        "stop_positive_weight": config.stop_positive_weight,
        "stop_positive_weight_max": config.stop_positive_weight_max,
    }


def _reduce_metrics(values: Sequence[float], context: DistributedContext) -> list[float]:
    tensor = torch.tensor(values, device=context.device, dtype=torch.float64)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return tensor.cpu().tolist()


def _clone_health_probe_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.detach().clone() if isinstance(value, torch.Tensor) else value)
        for key, value in inputs.items()
        if key != "generator"
    }


@torch.no_grad()
def _evaluate_health_probe(
    model: torch.nn.Module,
    inputs: Mapping[str, Any],
    config: TrainerConfig,
    context: DistributedContext,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=context.device)
    generator.manual_seed(config.seed + 20_000_000 + context.rank)
    probe_inputs = dict(inputs)
    probe_inputs["cfg_dropout_prob"] = 0.0
    probe_inputs["generator"] = generator
    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**probe_inputs)
    finally:
        model.train(was_training)
    values = _reduce_metrics(
        [
            float(output.loss),
            float(output.flow_loss),
            float(output.stop_loss),
            float(output.stop_positive_probability),
            float(output.stop_max_negative_probability),
            float(output.stop_logit_margin),
        ],
        context,
    )
    return dict(
        zip(
            (
                "loss",
                "flow_loss",
                "stop_loss",
                "stop_positive_probability",
                "stop_max_negative_probability",
                "stop_logit_margin",
            ),
            values,
            strict=True,
        )
    )


def _health_report_path(config: TrainerConfig) -> Path:
    return Path(config.output_dir) / f"TRAINING_HEALTH_STEP_{config.health_check_step:08d}.json"


def _loss_health_report_path(config: TrainerConfig) -> Path:
    return (
        Path(config.output_dir)
        / f"TRAINING_LOSS_HEALTH_STEP_{config.health_loss_check_step:08d}.json"
    )


def _run_training_health_check(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    monitor: ParameterChangeMonitor,
    probe_inputs: Mapping[str, Any],
    probe_before: Mapping[str, float],
    config: TrainerConfig,
    context: DistributedContext,
    global_step: int,
) -> dict[str, Any]:
    if global_step != config.health_check_step:
        raise ValueError("training health check called at the wrong step")
    optimizer_precision = _optimizer_precision_report(optimizer, context)
    parameter_delta = monitor.report(context)
    probe_after = _evaluate_health_probe(model, probe_inputs, config, context)
    baseline_flow = float(probe_before["flow_loss"])
    flow_relative_decrease = (baseline_flow - probe_after["flow_loss"]) / max(
        abs(baseline_flow), 1.0e-12
    )
    checks = {
        "optimizer_state_fp32": optimizer_precision["non_float32_values"] == 0,
        "parameters_changed": (
            parameter_delta["all"]["changed_fraction"]
            >= config.health_min_changed_fraction
        ),
        "parameter_relative_rms_delta": (
            parameter_delta["all"]["relative_rms_delta"]
            >= config.health_min_relative_rms_delta
        ),
        "stop_head_changed": (
            not config.health_require_stop_head_change
            or parameter_delta["stop_head"]["changed_values"] > 0
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    report = {
        "schema_version": 1,
        "status": "passed" if not failed else "failed",
        "step": global_step,
        "failed_checks": failed,
        "checks": checks,
        "thresholds": {
            "min_changed_fraction": config.health_min_changed_fraction,
            "min_relative_rms_delta": config.health_min_relative_rms_delta,
            "require_stop_head_change": config.health_require_stop_head_change,
        },
        "optimizer_precision": optimizer_precision,
        "parameter_delta": parameter_delta,
        "fixed_probe": {
            "before": dict(probe_before),
            "after": probe_after,
            "flow_loss_relative_decrease": flow_relative_decrease,
        },
    }
    if context.is_main:
        _atomic_json(_health_report_path(config), report)
    barrier()
    if failed:
        raise RuntimeError(f"training health gate failed at step {global_step}: {failed}")
    return report


def _run_training_loss_health_check(
    *,
    model: torch.nn.Module,
    probe_inputs: Mapping[str, Any],
    probe_before: Mapping[str, float],
    config: TrainerConfig,
    context: DistributedContext,
    global_step: int,
) -> dict[str, Any]:
    if global_step != config.health_loss_check_step:
        raise ValueError("training loss health check called at the wrong step")
    probe_after = _evaluate_health_probe(model, probe_inputs, config, context)
    baseline_flow = float(probe_before["flow_loss"])
    flow_relative_decrease = (baseline_flow - probe_after["flow_loss"]) / max(
        abs(baseline_flow), 1.0e-12
    )
    passed = (
        flow_relative_decrease
        > config.health_min_probe_flow_loss_relative_decrease
    )
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "step": global_step,
        "failed_checks": [] if passed else ["fixed_probe_flow_loss_decreased"],
        "checks": {"fixed_probe_flow_loss_decreased": passed},
        "thresholds": {
            "min_probe_flow_loss_relative_decrease": (
                config.health_min_probe_flow_loss_relative_decrease
            )
        },
        "fixed_probe": {
            "before": dict(probe_before),
            "after": probe_after,
            "flow_loss_relative_decrease": flow_relative_decrease,
        },
    }
    if context.is_main:
        _atomic_json(_loss_health_report_path(config), report)
    barrier()
    if not passed:
        raise RuntimeError(
            f"training loss health gate failed at step {global_step}: "
            f"flow_loss_relative_decrease={flow_relative_decrease}"
        )
    return report


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: Any,
    features: FrozenAudioFeatures,
    config: TrainerConfig,
    context: DistributedContext,
    global_step: int,
) -> dict[str, float]:
    model.eval()
    totals = torch.zeros(9, device=context.device, dtype=torch.float64)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(config.seed + 10_000_000)
    for index, batch in enumerate(loader):
        if index >= config.max_eval_samples:
            break
        inputs = _batch_to_inputs(batch, tokenizer, features, config, context)
        inputs["cfg_dropout_prob"] = 0.0
        inputs["generator"] = generator
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(**inputs)
        totals += torch.tensor(
            [
                float(output.loss),
                float(output.flow_loss),
                float(output.stop_loss),
                float(output.stop_positive_probability),
                float(output.stop_max_negative_probability),
                float(output.stop_logit_margin),
                float(output.stop_positive_probability >= 0.5),
                float(output.stop_max_negative_probability >= 0.5),
                1.0,
            ],
            device=context.device,
            dtype=torch.float64,
        )
    if context.distributed:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    count = max(float(totals[8]), 1.0)
    result = {
        "step": float(global_step),
        "eval_loss": float(totals[0] / count),
        "eval_flow_loss": float(totals[1] / count),
        "eval_stop_loss": float(totals[2] / count),
        "eval_stop_positive_probability": float(totals[3] / count),
        "eval_stop_max_negative_probability": float(totals[4] / count),
        "eval_stop_logit_margin": float(totals[5] / count),
        "eval_stop_recall_at_0_5": float(totals[6] / count),
        "eval_stop_false_positive_rate_at_0_5": float(totals[7] / count),
        "eval_samples": float(totals[8]),
    }
    model.train()
    return result


def _trainer_state_path(checkpoint: Path) -> Path:
    return checkpoint / "trainer_state.pt"


def _assert_data_order_resume_contract(state: Mapping[str, Any], config: TrainerConfig) -> None:
    """Never interpret an older RandomSampler offset as a v2 data position."""

    version = state.get("data_order_resume_version")
    if version != config.data_order_resume_version:
        raise RuntimeError(
            "checkpoint data-order contract is incompatible: "
            f"checkpoint={version!r}, trainer={config.data_order_resume_version!r}; "
            "use a fresh output directory/run, not a resume of a legacy checkpoint"
        )
    if not state.get("training_contract_sha256"):
        raise RuntimeError("epoch_seeded_v2 resume requires an immutable training contract SHA-256")


def _rank_state_path(checkpoint: Path, rank: int) -> Path:
    return checkpoint / f"rank_state-{rank:05d}.pt"


def _training_contract(config: TrainerConfig) -> dict[str, Any]:
    """Return the immutable config subset that must match across resumes."""

    value = asdict(config)
    for key in ("output_dir", "resume_from", "init_adapter_from"):
        value.pop(key, None)
    # Preserve hashes written before the precision/health fields existed when
    # those fields retain their disabled legacy defaults.  New runs that opt in
    # still bind the settings into their immutable resume contract.
    legacy_optional_defaults = {
        "optimizer_foreach": None,
        "text_sequence_format": "legacy",
        "prompt_language_tags": {},
        "stop_positive_weight_max": 64.0,
        "health_check_step": 0,
        "health_loss_check_step": 0,
        "health_parameter_samples_per_tensor": 32,
        "health_min_changed_fraction": 0.01,
        "health_min_relative_rms_delta": 1.0e-7,
        "health_min_probe_flow_loss_relative_decrease": 0.0,
        "health_require_stop_head_change": True,
    }
    for key, default in legacy_optional_defaults.items():
        if value.get(key) == default:
            value.pop(key, None)
    return value


def _training_contract_sha256(config: TrainerConfig) -> str:
    payload = json.dumps(
        _training_contract(config),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(value, temporary)
    temporary.replace(path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _snapshot_precision_report(
    output_dir: str | Path,
    checkpoint: Path,
) -> tuple[dict[str, Any], Path, str]:
    """Copy the mutable run-level precision proof into one immutable checkpoint."""
    source = _precision_report_path(output_dir)
    if not source.is_file():
        raise RuntimeError("precision verification report is missing before checkpoint save")
    payload = source.read_bytes()
    precision = json.loads(payload.decode("utf-8"))
    if precision.get("status") != "passed":
        raise RuntimeError(f"precision verification did not pass: {precision}")
    destination = checkpoint / "PRECISION.json"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return precision, destination, hashlib.sha256(payload).hexdigest()


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainerConfig,
    context: DistributedContext,
    strategy: str,
    adapter_config: LoRAConfig | None,
    global_step: int,
    epoch: int,
    batches_in_epoch: int,
    train_generator: torch.Generator,
) -> Path:
    checkpoint = Path(config.output_dir) / f"checkpoint-{global_step:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if strategy == "fsdp":
        from torch.distributed import checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_state_dict

        # Flatten optimizer state by parameter FQN.  PyTorch DCP otherwise
        # serializes the nested param_groups list as ``param_groups.0.*`` but a
        # fresh optimizer asks the load planner for the opaque
        # ``param_groups`` key, making an otherwise valid checkpoint unloadable.
        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            flatten_optimizer_state_dict=True,
        )
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        dcp.save(
            {"model": model_state, "optimizer": optimizer_state},
            checkpoint_id=checkpoint / "distcp",
        )
    elif context.is_main:
        if adapter_config is None:
            raise RuntimeError("LoRA checkpoint requested without adapter config")
        training_model = _unwrap_training_model(model)
        save_lora_adapter(
            training_model.core,
            checkpoint,
            adapter_config,
            metadata={"global_step": global_step, "language_tag": config.language_tag},
        )

    rank_state = {
        "rank": context.rank,
        "world_size": context.world_size,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(context.device),
        "train_generator_state": train_generator.get_state(),
    }
    _atomic_torch_save(rank_state, _rank_state_path(checkpoint, context.rank))
    barrier()

    if context.is_main:
        state = {
            "global_step": global_step,
            "data_order_resume_version": config.data_order_resume_version,
            "epoch": epoch,
            "batches_in_epoch": batches_in_epoch,
            "world_size": context.world_size,
            "training_contract_sha256": _training_contract_sha256(config),
            "scheduler": scheduler.state_dict(),
            "optimizer": (optimizer.state_dict() if strategy != "fsdp" else None),
            # Retained for compatibility with checkpoints written before rank
            # state files were introduced.
            "torch_rng_state": torch.get_rng_state(),
        }
        _atomic_torch_save(state, _trainer_state_path(checkpoint))
        # Precision and learning-health provenance is required for every new
        # checkpoint, including LoRA.  Keeping it only in FSDP checkpoints made
        # routed-adapter experiments impossible to promote through the same
        # fail-closed gate as full SFT.
        precision, precision_path, precision_sha256 = _snapshot_precision_report(
            config.output_dir,
            checkpoint,
        )
        health: dict[str, Any] | None = None
        if config.health_check_step and global_step >= config.health_check_step:
            health_path = _health_report_path(config)
            if not health_path.is_file():
                raise RuntimeError("configured training health report is missing")
            health_value = json.loads(health_path.read_text(encoding="utf-8"))
            if health_value.get("status") != "passed":
                raise RuntimeError(f"training health verification did not pass: {health_value}")
            health = {
                "path": str(health_path.resolve()),
                "sha256": hashlib.sha256(health_path.read_bytes()).hexdigest(),
                "status": "passed",
                "step": int(health_value["step"]),
            }
        loss_health: dict[str, Any] | None = None
        if config.health_loss_check_step and global_step >= config.health_loss_check_step:
            loss_health_path = _loss_health_report_path(config)
            if not loss_health_path.is_file():
                raise RuntimeError("configured training loss health report is missing")
            loss_health_value = json.loads(loss_health_path.read_text(encoding="utf-8"))
            if loss_health_value.get("status") != "passed":
                raise RuntimeError(
                    f"training loss health verification did not pass: {loss_health_value}"
                )
            loss_health = {
                "path": str(loss_health_path.resolve()),
                "sha256": hashlib.sha256(loss_health_path.read_bytes()).hexdigest(),
                "status": "passed",
                "step": int(loss_health_value["step"]),
            }
        _atomic_json(
            checkpoint / "metadata.json",
            {
                "schema_version": 3,
                "checkpoint_type": "full_fsdp" if strategy == "fsdp" else "lora",
                "data_order_resume_version": config.data_order_resume_version,
                "global_step": global_step,
                "epoch": epoch,
                "batches_in_epoch": batches_in_epoch,
                "world_size": context.world_size,
                "training_contract_sha256": _training_contract_sha256(config),
                "language_tag": config.language_tag,
                "text_sequence_format": config.text_sequence_format,
                "prompt_language_tags": config.prompt_language_tags,
                "initialization": {
                    "core_model_dir": config.core_model_dir,
                    "stage_parent_checkpoint": config.stage_parent_checkpoint,
                },
                "precision": {
                    "compute_dtype": config.dtype,
                    "trainable_master_dtype": precision["trainable_master_dtype"],
                    "optimizer_state_dtype": precision["optimizer_state_dtype"],
                    "report": str(precision_path.resolve()),
                    "report_sha256": precision_sha256,
                },
                "training_health": health,
                "training_loss_health": loss_health,
            },
        )
    barrier()
    return checkpoint


def load_training_state(
    checkpoint: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    context: DistributedContext,
    strategy: str,
    config: TrainerConfig,
    train_generator: torch.Generator,
) -> tuple[int, int, int]:
    checkpoint = Path(checkpoint)
    if strategy == "fsdp":
        from torch.distributed import checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_state_dict,
            set_state_dict,
        )

        options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
            flatten_optimizer_state_dict=True,
        )
        model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
        state = {"model": model_state, "optimizer": optimizer_state}
        dcp.load(state, checkpoint_id=checkpoint / "distcp")
        set_state_dict(
            model,
            optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
            options=options,
        )
    trainer_state = torch.load(_trainer_state_path(checkpoint), map_location="cpu", weights_only=False)
    _assert_data_order_resume_contract(trainer_state, config)
    expected_contract = trainer_state.get("training_contract_sha256")
    actual_contract = _training_contract_sha256(config)
    if expected_contract is not None and expected_contract != actual_contract:
        raise RuntimeError(
            "resume config does not match checkpoint training contract: "
            f"checkpoint={expected_contract}, current={actual_contract}"
        )
    checkpoint_world_size = int(trainer_state.get("world_size", context.world_size))
    if checkpoint_world_size != context.world_size:
        raise RuntimeError(
            "exact data-order resume requires unchanged world size: "
            f"checkpoint={checkpoint_world_size}, current={context.world_size}"
        )
    if strategy != "fsdp":
        optimizer.load_state_dict(trainer_state["optimizer"])
    scheduler.load_state_dict(trainer_state["scheduler"])
    rank_path = _rank_state_path(checkpoint, context.rank)
    if rank_path.is_file():
        rank_state = torch.load(rank_path, map_location="cpu", weights_only=False)
        if int(rank_state["rank"]) != context.rank:
            raise RuntimeError(f"rank state mismatch in {rank_path}")
        if int(rank_state["world_size"]) != context.world_size:
            raise RuntimeError(f"rank-state world size mismatch in {rank_path}")
        random.setstate(rank_state["python_rng_state"])
        torch.set_rng_state(rank_state["torch_rng_state"])
        torch.cuda.set_rng_state(rank_state["cuda_rng_state"], context.device)
        train_generator.set_state(rank_state["train_generator_state"])
    elif trainer_state.get("torch_rng_state") is not None:
        # Backward compatibility is intentionally weaker: old checkpoints did
        # not preserve per-rank CUDA or flow-generator state.
        torch.set_rng_state(trainer_state["torch_rng_state"])
    return (
        int(trainer_state["global_step"]),
        int(trainer_state.get("epoch", 0)),
        int(trainer_state.get("batches_in_epoch", 0)),
    )


def _prune_checkpoints(config: TrainerConfig, context: DistributedContext) -> None:
    if not context.is_main or config.keep_last_checkpoints < 0:
        return
    checkpoints = sorted(Path(config.output_dir).glob("checkpoint-*"))
    for path in checkpoints[: max(0, len(checkpoints) - config.keep_last_checkpoints)]:
        # Checkpoints are created solely by this trainer under output_dir.
        import shutil

        shutil.rmtree(path)


def _write_run_metadata(
    config: TrainerConfig,
    train_dataset: PromptPairedDataset,
    valid_dataset: PromptPairedDataset | None,
    context: DistributedContext,
    stop_at_step: int,
) -> None:
    if not context.is_main:
        return
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resolved_path = output / "config.resolved.json"
    resolved_payload = json.dumps(
        asdict(config), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    if resolved_path.is_file():
        previous_payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        if previous_payload.get("data_order_resume_version") != config.data_order_resume_version:
            raise RuntimeError("refusing to reuse a legacy run directory under the v2 data-order contract")
        previous = TrainerConfig(**previous_payload)
        if _training_contract_sha256(previous) != _training_contract_sha256(config):
            raise RuntimeError(
                f"refusing to mix incompatible training configs in {config.output_dir}"
            )
    else:
        resolved_path.write_text(resolved_payload, encoding="utf-8")
    summary = {
        "train": dataset_summary(train_dataset),
        "validation": (None if valid_dataset is None else dataset_summary(valid_dataset)),
    }
    summary_path = output / "data_summary.json"
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if summary_path.is_file() and summary_path.read_text(encoding="utf-8") != summary_payload:
        raise RuntimeError(f"training data summary changed in {config.output_dir}")
    if not summary_path.is_file():
        summary_path.write_text(summary_payload, encoding="utf-8")
    launch = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "world_size": context.world_size,
        "resume_from": config.resume_from,
        "init_adapter_from": config.init_adapter_from,
        "core_model_dir": config.core_model_dir,
        "stage_parent_checkpoint": config.stage_parent_checkpoint,
        "planned_max_steps": config.max_steps,
        "stop_at_step": stop_at_step,
        "training_contract_sha256": _training_contract_sha256(config),
    }
    with (output / "launch_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(launch, ensure_ascii=False, sort_keys=True) + "\n")


def _append_metrics(output_dir: str, values: Mapping[str, Any], context: DistributedContext) -> None:
    if not context.is_main:
        return
    with (Path(output_dir) / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(values), ensure_ascii=False, sort_keys=True) + "\n")


def run_training(
    config: TrainerConfig,
    *,
    audit_only: bool = False,
    stop_at_step: int | None = None,
) -> None:
    config.validate()
    training_stop_step = config.max_steps if stop_at_step is None else int(stop_at_step)
    if training_stop_step <= 0 or training_stop_step > config.max_steps:
        raise ValueError("stop_at_step must be in [1, config.max_steps]")
    # Auditing deliberately runs before process-group/model initialization.
    train_dataset, valid_dataset = _build_datasets(config)
    if audit_only:
        print(json.dumps({
            "planned_max_steps": config.max_steps,
            "scheduler_steps": config.scheduler_steps or config.max_steps,
            "stop_at_step": training_stop_step,
            "train": dataset_summary(train_dataset),
            "validation": None if valid_dataset is None else dataset_summary(valid_dataset),
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return

    context = init_distributed(config.strategy)
    if config.expected_world_size is not None and context.world_size != config.expected_world_size:
        raise RuntimeError(
            "distributed world size changed: "
            f"expected={config.expected_world_size}, actual={context.world_size}"
        )
    seed_everything(config.seed, context.rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _write_run_metadata(config, train_dataset, valid_dataset, context, training_stop_step)
    log(f"train data: {json.dumps(dataset_summary(train_dataset), ensure_ascii=False)}", context)
    if valid_dataset is not None:
        log(f"valid data: {json.dumps(dataset_summary(valid_dataset), ensure_ascii=False)}", context)

    train_loader, valid_loader, train_sampler, _ = _build_loaders(
        config, context, train_dataset, valid_dataset
    )
    if (
        config.expected_steps_per_epoch is not None
        and len(train_loader) != config.expected_steps_per_epoch
    ):
        raise RuntimeError(
            "steps per epoch changed: "
            f"expected={config.expected_steps_per_epoch}, actual={len(train_loader)}"
        )
    tokenizer = load_text_tokenizer(str(Path(config.pretrained_model_dir) / "text_tokenizer"))
    training_model, adapter_config = _load_core(config, context)
    embedding_rows = training_model.core.backbone_llm.embed_tokens.num_embeddings
    if len(tokenizer) > embedding_rows:
        raise RuntimeError(
            f"tokenizer has {len(tokenizer)} entries but model has {embedding_rows} embedding rows"
        )
    language_token = f"<|{config.language_tag}|>"
    language_ids = tokenizer(language_token, add_special_tokens=False)["input_ids"]
    if len(language_ids) != 1:
        raise RuntimeError(f"language tag must be one token, got {language_ids} for {language_token}")
    log(f"language token {language_token} -> id {language_ids[0]}", context)

    patch_size = int(training_model.core.patch_size)
    model, strategy = _wrap_model(training_model, config, context)
    parameter_precision = _assert_trainable_parameters_fp32(model, context)
    optimizer = _build_optimizer(model, config)
    scheduler = _build_scheduler(optimizer, config)
    features = _load_frozen_features(config, context, patch_size)
    generator = torch.Generator(device=context.device)
    generator.manual_seed(config.seed + context.rank)
    global_step = 0
    epoch = 0
    batches_in_epoch = 0
    if config.resume_from:
        global_step, epoch, batches_in_epoch = load_training_state(
            config.resume_from,
            model,
            optimizer,
            scheduler,
            context,
            strategy,
            config,
            generator,
        )
        log(
            f"resumed {config.resume_from} at step={global_step} "
            f"epoch={epoch} batches_in_epoch={batches_in_epoch}",
            context,
        )
        optimizer_precision = _optimizer_precision_report(optimizer, context)
        _write_precision_report(
            config,
            parameter_precision,
            optimizer_precision,
            context,
            verified_at_step=global_step,
        )
    elif config.init_adapter_from:
        log(
            f"initialized adapter weights from {config.init_adapter_from}; "
            "optimizer/scheduler/global_step reset",
            context,
        )

    health_monitor: ParameterChangeMonitor | None = None
    health_probe_inputs: dict[str, Any] | None = None
    health_probe_before: dict[str, float] | None = None
    if max(config.health_check_step, config.health_loss_check_step) > global_step:
        health_monitor = ParameterChangeMonitor(
            model, config.health_parameter_samples_per_tensor
        )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    train_dataset.set_epoch(epoch)
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(epoch)
    iterator = iter(train_loader)
    if batches_in_epoch > len(train_loader):
        raise RuntimeError(
            f"checkpoint batches_in_epoch={batches_in_epoch} exceeds loader length={len(train_loader)}"
        )
    for _ in range(batches_in_epoch):
        try:
            next(iterator)
        except StopIteration as error:
            raise RuntimeError("checkpoint data position cannot be restored") from error
    if batches_in_epoch:
        log(f"restored exact data position after {batches_in_epoch} batches", context)
    accumulation = []
    micro_step = 0
    start_time = time.monotonic()
    last_saved_step = -1

    while global_step < training_stop_step:
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            batches_in_epoch = 0
            train_dataset.set_epoch(epoch)
            if hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)
            iterator = iter(train_loader)
            batch = next(iterator)
        batches_in_epoch += 1

        inputs = _batch_to_inputs(batch, tokenizer, features, config, context)
        if health_monitor is not None and health_probe_inputs is None:
            health_probe_inputs = _clone_health_probe_inputs(inputs)
            health_probe_before = _evaluate_health_probe(
                model, health_probe_inputs, config, context
            )
            log(
                f"fixed training-health probe baseline: {json.dumps(health_probe_before, sort_keys=True)}",
                context,
            )
        inputs["generator"] = generator
        micro_step += 1
        should_sync = micro_step % config.gradient_accumulation_steps == 0
        sync_context = (
            contextlib.nullcontext()
            if should_sync or not hasattr(model, "no_sync")
            else model.no_sync()
        )
        with sync_context:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(**inputs)
                scaled_loss = output.loss / config.gradient_accumulation_steps
            if config.fail_on_nonfinite:
                finite = torch.stack(
                    [
                        torch.isfinite(output.loss),
                        torch.isfinite(output.flow_loss),
                        torch.isfinite(output.stop_loss),
                    ]
                ).all().to(dtype=torch.int32)
                if context.distributed:
                    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not bool(finite.item()):
                    raise FloatingPointError(
                        f"non-finite loss at global_step={global_step} rank={context.rank}"
                    )
            scaled_loss.backward()
        accumulation.append(
            (
                float(output.loss.detach()),
                float(output.flow_loss.detach()),
                float(output.stop_loss.detach()),
                float(output.stop_positive_probability.detach()),
                float(output.stop_max_negative_probability.detach()),
                float(output.stop_logit_margin.detach()),
                float(output.effective_stop_positive_weight),
            )
        )
        if not should_sync:
            continue

        if strategy == "fsdp":
            grad_norm = model.clip_grad_norm_(config.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                config.max_grad_norm,
            )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

        if global_step == 1:
            optimizer_precision = _optimizer_precision_report(optimizer, context)
            _write_precision_report(
                config,
                parameter_precision,
                optimizer_precision,
                context,
                verified_at_step=global_step,
            )

        if config.health_check_step and global_step == config.health_check_step:
            if (
                health_monitor is None
                or health_probe_inputs is None
                or health_probe_before is None
            ):
                raise RuntimeError("training health monitor was not initialized")
            health = _run_training_health_check(
                model=model,
                optimizer=optimizer,
                monitor=health_monitor,
                probe_inputs=health_probe_inputs,
                probe_before=health_probe_before,
                config=config,
                context=context,
                global_step=global_step,
            )
            log(f"training health passed: {json.dumps(health, sort_keys=True)}", context)

        if (
            config.health_loss_check_step
            and global_step == config.health_loss_check_step
        ):
            if health_probe_inputs is None or health_probe_before is None:
                raise RuntimeError("training loss health probe was not initialized")
            loss_health = _run_training_loss_health_check(
                model=model,
                probe_inputs=health_probe_inputs,
                probe_before=health_probe_before,
                config=config,
                context=context,
                global_step=global_step,
            )
            log(
                f"training loss health passed: {json.dumps(loss_health, sort_keys=True)}",
                context,
            )

        local_means = [
            sum(row[index] for row in accumulation) / len(accumulation)
            for index in range(7)
        ]
        accumulation.clear()
        if global_step % config.log_steps == 0 or global_step == 1:
            (
                loss,
                flow_loss,
                stop_loss,
                stop_positive_probability,
                stop_max_negative_probability,
                stop_logit_margin,
                effective_stop_positive_weight,
            ) = _reduce_metrics(local_means, context)
            elapsed = max(time.monotonic() - start_time, 1e-6)
            metrics = {
                "step": global_step,
                "epoch": epoch,
                "loss": loss,
                "flow_loss": flow_loss,
                "stop_loss": stop_loss,
                "stop_positive_probability": stop_positive_probability,
                "stop_max_negative_probability": stop_max_negative_probability,
                "stop_logit_margin": stop_logit_margin,
                "effective_stop_positive_weight": effective_stop_positive_weight,
                "learning_rate": scheduler.get_last_lr()[0],
                "grad_norm": float(grad_norm),
                "steps_per_second": global_step / elapsed,
                "elapsed_training_seconds": elapsed,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(context.device) / (1024 ** 2),
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(context.device) / (1024 ** 2),
            }
            log(json.dumps(metrics, sort_keys=True), context)
            _append_metrics(config.output_dir, metrics, context)

        if valid_loader is not None and config.eval_steps > 0 and global_step % config.eval_steps == 0:
            metrics = evaluate(
                model, valid_loader, tokenizer, features, config, context, global_step
            )
            log(json.dumps(metrics, sort_keys=True), context)
            _append_metrics(config.output_dir, metrics, context)

        periodic_save = config.save_steps > 0 and global_step % config.save_steps == 0
        explicit_save = global_step in config.save_at_steps
        if periodic_save or explicit_save:
            path = save_checkpoint(
                model,
                optimizer,
                scheduler,
                config,
                context,
                strategy,
                adapter_config,
                global_step,
                epoch,
                batches_in_epoch,
                generator,
            )
            last_saved_step = global_step
            log(f"saved {path}", context)
            _prune_checkpoints(config, context)
            barrier()

    if last_saved_step != global_step:
        path = save_checkpoint(
            model,
            optimizer,
            scheduler,
            config,
            context,
            strategy,
            adapter_config,
            global_step,
            epoch,
            batches_in_epoch,
            generator,
        )
        log(f"saved final {path}", context)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="JSON training config")
    parser.add_argument("--audit-only", action="store_true", help="validate manifests/pairs and exit")
    parser.add_argument("--resume-from", help="override resume_from in the JSON config")
    parser.add_argument(
        "--init-adapter-from",
        help="initialize LoRA weights but reset optimizer, scheduler, and global step",
    )
    parser.add_argument("--output-dir", help="override output_dir in the JSON config")
    parser.add_argument("--max-steps", type=int, help="override max_steps (useful for smoke tests)")
    parser.add_argument(
        "--stop-at-step",
        type=int,
        help="stop this launch early without changing the planned LR schedule or run contract",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_argument_parser().parse_args(argv)
    config = TrainerConfig.from_json(args.config)
    if args.resume_from:
        config.resume_from = args.resume_from
        # Resuming restores the already-initialized adapter plus optimizer and
        # scheduler state.  It therefore supersedes a stage config's cold-start
        # adapter initializer (which is still needed for the original launch).
        config.init_adapter_from = None
    if args.init_adapter_from:
        config.init_adapter_from = args.init_adapter_from
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    run_training(config, audit_only=args.audit_only, stop_at_step=args.stop_at_step)


if __name__ == "__main__":
    main()
