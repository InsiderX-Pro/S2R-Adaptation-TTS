#!/usr/bin/env python3
"""Consolidate a FireRed full-FSDP DCP checkpoint into an inference core."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import save_file

from fireredtts3.llm.fireredtts3_base import FireRedTTS3BaseCoreConfig
from fireredtts3.training.trainer import (
    TrainerConfig,
    _load_core,
    _training_contract_sha256,
    _wrap_model,
    barrier,
    init_distributed,
    seed_everything,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_source(checkpoint: Path, config: TrainerConfig, expected_step: int) -> dict[str, object]:
    required = [
        checkpoint / "distcp" / ".metadata",
        checkpoint / "trainer_state.pt",
        checkpoint / "metadata.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"incomplete FSDP checkpoint; missing={missing}")
    metadata = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("checkpoint_type") != "full_fsdp":
        raise RuntimeError(f"not a full-FSDP checkpoint: {metadata}")
    if int(metadata.get("global_step", -1)) != expected_step:
        raise RuntimeError(f"checkpoint step mismatch: expected={expected_step}, metadata={metadata}")
    if int(metadata.get("world_size", -1)) != int(config.expected_world_size or -1):
        raise RuntimeError(f"checkpoint world-size mismatch: {metadata}")
    contract = _training_contract_sha256(config)
    if metadata.get("training_contract_sha256") != contract:
        raise RuntimeError("checkpoint training contract does not match export config")
    if config.health_check_step and expected_step >= config.health_check_step:
        precision = metadata.get("precision", {})
        health = metadata.get("training_health", {})
        if (
            precision.get("trainable_master_dtype") != "float32"
            or precision.get("optimizer_state_dtype") != "float32"
        ):
            raise RuntimeError(f"checkpoint lacks verified FP32 training precision: {metadata}")
        if (
            not isinstance(health, dict)
            or health.get("status") != "passed"
            or int(health.get("step", -1)) != config.health_check_step
        ):
            raise RuntimeError(f"checkpoint lacks the configured training health gate: {metadata}")
    if config.health_loss_check_step and expected_step >= config.health_loss_check_step:
        loss_health = metadata.get("training_loss_health", {})
        if (
            not isinstance(loss_health, dict)
            or loss_health.get("status") != "passed"
            or int(loss_health.get("step", -1)) != config.health_loss_check_step
        ):
            raise RuntimeError(
                f"checkpoint lacks the configured training loss health gate: {metadata}"
            )
    return metadata


def strip_training_wrapper(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    prefixes = ("core.", "_fsdp_wrapped_module.core.", "module.core.")
    for prefix in prefixes:
        selected = {
            key[len(prefix) :]: tensor.detach().cpu().contiguous()
            for key, tensor in state.items()
            if key.startswith(prefix)
        }
        if selected and len(selected) == len(state):
            return selected
    sample = sorted(state)[:20]
    raise RuntimeError(f"unexpected full state-dict keys; sample={sample}")


def cast_core_state_for_inference(
    state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Quantize the FP32 training master weights only at export time."""

    return {
        name: (
            tensor.to(torch.bfloat16).contiguous()
            if tensor.is_floating_point()
            else tensor.contiguous()
        )
        for name, tensor in state.items()
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    config_path = args.config.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    config = TrainerConfig.from_json(config_path)
    config.validate()
    if config.mode != "full" or config.strategy != "fsdp":
        raise ValueError("export requires mode=full and strategy=fsdp")
    source_metadata = validate_source(checkpoint, config, args.expected_step)

    context = init_distributed(config.strategy)
    if config.expected_world_size is not None and context.world_size != config.expected_world_size:
        raise RuntimeError(
            f"export world size must be {config.expected_world_size}, got {context.world_size}"
        )
    seed_everything(config.seed, context.rank)
    training_model, _ = _load_core(config, context)
    model, strategy = _wrap_model(training_model, config, context)
    if strategy != "fsdp":
        raise AssertionError(strategy)

    from torch.distributed import checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )
    from torch.distributed.fsdp import (
        FullStateDictConfig,
        FullyShardedDataParallel as FSDP,
        StateDictType,
    )

    sharded_options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=sharded_options)
    state = {"model": model_state}
    dcp.load(state, checkpoint_id=checkpoint / "distcp")
    incompatible = set_model_state_dict(model, state["model"], options=sharded_options)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"incompatible checkpoint: {incompatible}")
    barrier()

    full_options = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_options):
        full_training_state = model.state_dict()

    if context.is_main:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to replace non-empty export directory: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        master_core_state = strip_training_wrapper(full_training_state)
        if any(
            tensor.is_floating_point() and tensor.dtype != torch.float32
            for tensor in master_core_state.values()
        ):
            bad = sorted(
                f"{name}:{tensor.dtype}"
                for name, tensor in master_core_state.items()
                if tensor.is_floating_point() and tensor.dtype != torch.float32
            )[:20]
            raise RuntimeError(f"full-training state is not FP32 before export: {bad}")
        core_state = cast_core_state_for_inference(master_core_state)
        if not core_state or not all(torch.isfinite(tensor).all() for tensor in core_state.values()):
            raise RuntimeError("exported core state is empty or contains NaN/Inf")
        weights_path = output_dir / "model.safetensors"
        temporary_weights = weights_path.with_name(weights_path.name + f".tmp.{os.getpid()}")
        save_file(core_state, temporary_weights, metadata={"format": "pt"})
        os.replace(temporary_weights, weights_path)

        core_config = FireRedTTS3BaseCoreConfig.from_pretrained(
            str(Path(config.pretrained_model_dir) / "fireredtts3_base")
        )
        core_config.torch_dtype = "bfloat16"
        core_config.save_pretrained(output_dir)
        config_output = output_dir / "config.json"
        export_metadata = {
            "schema_version": 1,
            "status": "passed",
            "checkpoint_type": "full_fsdp_export",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_checkpoint": str(checkpoint),
            "source_checkpoint_metadata": source_metadata,
            "global_step": args.expected_step,
            "world_size": context.world_size,
            "training_contract_sha256": _training_contract_sha256(config),
            "training_precision": source_metadata.get("precision"),
            "training_health": source_metadata.get("training_health"),
            "training_loss_health": source_metadata.get("training_loss_health"),
            "parameter_tensors": len(core_state),
            "parameter_values": sum(tensor.numel() for tensor in core_state.values()),
            "weights": {
                "path": str(weights_path),
                "bytes": weights_path.stat().st_size,
                "sha256": sha256(weights_path),
                "dtype": "bfloat16",
            },
            "config": {
                "path": str(config_output),
                "sha256": sha256(config_output),
            },
        }
        atomic_json(output_dir / "export_metadata.json", export_metadata)
        print(json.dumps(export_metadata, ensure_ascii=False, sort_keys=True), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
