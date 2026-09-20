"""Single-device OmniVoice full/LoRA training from paired codec data, S then R."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile

from .backbones import external_output, use_backbone
from .firered import RECIPES, verify_manifest
from .manifests import sha256


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def model_fingerprint(directory):
    directory = Path(directory)
    files = sorted(directory.glob("*.safetensors"))
    if not files or not (directory / "config.json").is_file():
        raise FileNotFoundError("pretrained config.json and safetensors weights are required")
    return canonical_hash({p.name: sha256(p) for p in [directory / "config.json", *files]})


def configuration(manifest, tokens, pretrained, output_dir, *, language="my", seed=42,
                  mode="full", init_from=None, template=None):
    report = verify_manifest(manifest)
    stage = report["stage"]
    if (stage == "R") != bool(init_from):
        raise ValueError("R requires a completed S checkpoint; S initializes from Base")
    if template is None:
        template = Path(__file__).resolve().parent / "configs" / "omnivoice" / f"{language}_{stage}.json"
    config = json.loads(Path(template).read_text())
    config.update(stage=stage, language=language, seed=seed, mode=mode,
                  manifest=str(Path(manifest).resolve()), tokens=str(Path(tokens).resolve()),
                  pretrained=str(Path(pretrained).resolve()), output_dir=str(external_output(output_dir)),
                  init_from=str(Path(init_from).resolve()) if init_from else None)
    validate_config(config)
    return config


def validate_config(config):
    required = {"stage", "mode", "manifest", "tokens", "pretrained", "output_dir", "seed",
                "max_steps", "gradient_accumulation_steps", "learning_rate", "warmup_ratio",
                "weight_decay", "max_grad_norm", "save_steps", "log_steps", "device", "dtype",
                "gradient_checkpointing", "mask_ratio_range", "drop_cond_ratio", "init_from"}
    allowed = required | {"language", "adam_betas", "adam_eps", "lora"}
    if required - config.keys() or config.keys() - allowed:
        raise ValueError(f"config fields: missing={required-config.keys()}, unknown={config.keys()-allowed}")
    if config["stage"] not in {"S", "R"} or config["mode"] not in {"full", "lora"}:
        raise ValueError("stage must be S/R and mode must be full/lora")
    if (config["stage"] == "R") != bool(config["init_from"]):
        raise ValueError("S starts from Base; R starts from S")
    for key in ("max_steps", "gradient_accumulation_steps", "save_steps", "log_steps"):
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= config["warmup_ratio"] < 1 or not 0 <= config["drop_cond_ratio"] < 1:
        raise ValueError("invalid warmup/conditioning dropout")
    lo, hi = config["mask_ratio_range"]
    if not 0 <= lo <= hi <= 1:
        raise ValueError("invalid mask range")
    if config["dtype"] not in {"bfloat16", "float32"}:
        raise ValueError("dtype must be bfloat16 or float32")
    if not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("invalid learning rate")
    external_output(config["output_dir"])


def checkpoint_metadata(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "COMPLETE.json").read_text())
    for name, digest in metadata["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha256(path) != digest:
            raise ValueError("checkpoint checksum mismatch")
    return metadata


def _create_model(config, checkpoint, base_fingerprint):
    import torch
    from omnivoice.models.omnivoice import OmniVoice
    from .omni_lora import PeftSpec, inject_peft, load_adapter_checkpoint
    if checkpoint:
        previous = checkpoint_metadata(checkpoint)
        if previous["mode"] != config["mode"] or previous["base_fingerprint"] != base_fingerprint:
            raise ValueError("checkpoint mode/base model mismatch")
        path = Path(checkpoint) / "model" if config["mode"] == "full" else Path(config["pretrained"])
    else:
        path = Path(config["pretrained"])
    model = OmniVoice.from_pretrained(str(path), train=True, dtype=torch.float32,
                                     attn_implementation="sdpa", local_files_only=True)
    model.llm.config.use_cache = False
    if config["mode"] == "lora":
        if checkpoint:
            load_adapter_checkpoint(model, Path(checkpoint) / "adapter", trainable=True)
            saved = model._local_peft_spec.to_dict()
            if any(saved.get(key) != value for key, value in config["lora"].items()):
                raise ValueError("R/resume adapter settings differ from the saved S adapter")
        else:
            settings = dict(config["lora"])
            if settings.get("initialization", "vanilla") != "vanilla":
                raise ValueError("the S2R entry supports vanilla LoRA initialization")
            spec = PeftSpec.from_dict({"experiment_id": "P2_S2R_REFERENCE", "strategy": "lora",
                                      "base_model": str(config["pretrained"]), **settings})
            inject_peft(model, spec)
    if config["gradient_checkpointing"]:
        model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


def _save_checkpoint(directory, model, optimizer, scheduler, config, fingerprint, step, epoch, offset):
    import torch
    from .omni_lora import save_adapter_checkpoint
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=directory.parent))
    try:
        if config["mode"] == "full":
            model.save_pretrained(temporary / "model", safe_serialization=True)
        else:
            save_adapter_checkpoint(model, temporary / "adapter", model._local_peft_spec)
        state = {"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
                 "step": step, "epoch": epoch, "offset": offset}
        torch.save(state, temporary / "training_state.pt")
        (temporary / "config.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
        metadata = {"schema_version": 1, "stage": config["stage"], "mode": config["mode"],
                    "global_step": step, "base_fingerprint": fingerprint,
                    "config_sha256": canonical_hash(config), "files": {}}
        for file in sorted(temporary.rglob("*")):
            if file.is_file():
                metadata["files"][file.relative_to(temporary).as_posix()] = sha256(file)
        (temporary / "COMPLETE.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def train(config, *, resume=None):
    validate_config(config)
    if int(__import__("os").environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("this paper reference uses one device; distributed sharding is not enabled")
    use_backbone("omnivoice")
    import torch
    from transformers import AutoTokenizer
    from omnivoice.data.collator import PaddingDataCollator
    from .omni_data import PairedCodecDataset, PairedProcessor, collate_pairs
    from .omni_lora import inference_routing_mask, set_routing_mask
    from .losses import omnivoice_loss

    dataset = PairedCodecDataset(config["tokens"], config["manifest"])
    if dataset.metadata["stage"] != config["stage"] or not len(dataset):
        raise ValueError("empty or wrong-stage codec dataset")
    output = external_output(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError("fresh stages require an empty output directory")
    fingerprint = model_fingerprint(config["pretrained"])
    initial = resume or config["init_from"]
    if initial:
        previous = checkpoint_metadata(initial)
        if resume:
            if previous["config_sha256"] != canonical_hash(config):
                raise ValueError("resume requires exactly the same stage configuration")
        elif previous["stage"] != "S":
            raise ValueError("R must initialize from an S checkpoint")
        elif previous["global_step"] != json.loads((Path(initial) / "config.json").read_text())["max_steps"]:
            raise ValueError("R requires the completed S training budget")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config["seed"])
    model = _create_model(config, initial, fingerprint).to(device)
    if dataset.metadata["num_channels"] != model.config.num_audio_codebook or \
            dataset.metadata["vocab_size"] != model.config.audio_mask_id:
        raise ValueError("prepared codec and acoustic model vocabulary differ")
    tokenizer = AutoTokenizer.from_pretrained(config["pretrained"], local_files_only=True)
    processor = PairedProcessor(tokenizer, model.config.num_audio_codebook, model.config.audio_mask_id,
        mask_ratio_range=config["mask_ratio_range"], drop_cond_ratio=config["drop_cond_ratio"])
    collator = PaddingDataCollator(processor, batch_tokens=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
        betas=tuple(config.get("adam_betas", [0.9, 0.95])), eps=config.get("adam_eps", 1e-8),
        weight_decay=config["weight_decay"], foreach=False)
    warmup = math.ceil(config["max_steps"] * config["warmup_ratio"])

    def schedule(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, config["max_steps"] - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    step, epoch, offset = 0, 0, 0
    initial_optimizer_entries = len(optimizer.state)
    if resume:
        # Only consume the caller's own local checkpoint, verified above.
        state = torch.load(Path(resume) / "training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, epoch, offset = state["step"], state["epoch"], state["offset"]
        initial_optimizer_entries = len(optimizer.state)
    initial_step = step
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.json").write_text(json.dumps(config, indent=2)+"\n", encoding="utf-8")
    model.train()
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]+epoch)).tolist()
    dtype = torch.bfloat16 if config["dtype"] == "bfloat16" else torch.float32
    checkpoint = None
    while step < config["max_steps"]:
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config["gradient_accumulation_steps"]):
            if offset == len(order):
                epoch, offset = epoch + 1, 0
                order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(config["seed"]+epoch)).tolist()
            batch = collate_pairs([processor(dataset[order[offset]])], collator)
            offset += 1
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            labels, weights = batch.pop("labels"), batch.pop("sample_weights")
            if config["mode"] == "lora":
                routing = model._local_peft_spec.routing
                if routing == "target_region":
                    raise ValueError("use none/prediction/text_only/text_prediction routing in the paired trainer")
                set_routing_mask(model, inference_routing_mask(routing=routing,
                    input_ids=batch["input_ids"], audio_mask=batch["audio_mask"], audio_mask_id=model.config.audio_mask_id))
            precision = torch.autocast(device_type=device.type, dtype=dtype) if dtype != torch.float32 else contextlib.nullcontext()
            with precision:
                logits = model(**batch, labels=None).logits
                loss = omnivoice_loss(logits, labels, weights, model.normalized_audio_codebook_weights)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            (loss / config["gradient_accumulation_steps"]).backward()
            loss_sum += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(parameters, config["max_grad_norm"], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        step += 1
        if step % config["log_steps"] == 0 or step == config["max_steps"]:
            metric = {"step": step, "loss": loss_sum / config["gradient_accumulation_steps"],
                      "learning_rate": scheduler.get_last_lr()[0]}
            with (output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metric)+"\n")
            print(json.dumps(metric), flush=True)
        if step % config["save_steps"] == 0 or step == config["max_steps"]:
            checkpoint = output / f"checkpoint-{step:08d}"
            _save_checkpoint(checkpoint, model, optimizer, scheduler, config, fingerprint, step, epoch, offset)
    result = {"status": "completed", "stage": config["stage"], "mode": config["mode"], "global_step": step,
              "initial_step": initial_step, "initial_optimizer_entries": initial_optimizer_entries,
              "checkpoint": str(checkpoint or Path(resume)), "fresh_stage": resume is None}
    (output / "training_summary.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    configure = sub.add_parser("configure")
    for name in ("manifest", "tokens", "pretrained", "output-dir", "config-out"):
        configure.add_argument("--"+name, required=True)
    configure.add_argument("--language", choices=list(RECIPES), default="my")
    configure.add_argument("--seed", type=int, default=42)
    configure.add_argument("--mode", choices=["full", "lora"], default="full")
    configure.add_argument("--s-checkpoint")
    configure.add_argument("--template")
    run = sub.add_parser("train")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", help="Resume this same stage; never used for S to R")
    args = parser.parse_args(argv)
    if args.command == "configure":
        value = configuration(args.manifest, args.tokens, args.pretrained, args.output_dir,
            language=args.language, seed=args.seed, mode=args.mode, init_from=args.s_checkpoint, template=args.template)
        destination = external_output(args.config_out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
        print(destination)
    else:
        print(json.dumps(train(json.loads(Path(args.config).read_text()), resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
