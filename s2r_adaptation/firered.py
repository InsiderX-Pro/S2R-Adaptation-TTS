"""Configure, validate and train with the included FireRed training extension."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

from .agreement import reliability_weight
from .manifests import fields, read_rows, row_id, sha256
from .backbones import use_backbone, external_output

RECIPES = {"my": {"S": 53640, "R": 16015}, "lo": {"S": 39749, "R": 8668}}


def verify_manifest(manifest):
    manifest = Path(manifest)
    report = json.loads((manifest.parent / "report.json").read_text())
    if sha256(manifest) != report["train_sha256"]:
        raise ValueError("prepared training manifest checksum mismatch")
    count = 0
    for row in read_rows(manifest):
        count += 1
        evidence = row["reliability"]
        if evidence["stage"] != report["stage"]:
            raise ValueError("manifest mixes S/R stages")
        expected = 1.0 if report["stage"] == "S" else reliability_weight(
            evidence["disagreement"], report["gamma"], report["w_min"])
        label = fields(row)
        if not math.isclose(label["pseudo_label_weight"], expected, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError("manifest sample weight violates declared formula")
        if "loss_weight" in label and label["loss_weight"] != label["pseudo_label_weight"]:
            raise ValueError("conflicting sample weight aliases")
    if count != report["accepted_rows"]:
        raise ValueError("prepared manifest count mismatch")
    return report


def stage_config(template, manifest, output_dir, pretrained, *, language="my", seed=42, s_core=None):
    report = verify_manifest(manifest)
    stage = report["stage"]
    config = copy.deepcopy(template)
    if language not in RECIPES:
        raise ValueError("paper presets support my and lo")
    if (stage == "R") != (s_core is not None):
        raise ValueError("R requires the S model export; S must start from Base")
    steps = RECIPES[language][stage]
    source_name = config.get("train_sources", [{}])[0].get("name", "s2r")
    config.update(pretrained_model_dir=str(Path(pretrained).resolve()),
        output_dir=str(Path(output_dir).resolve()), language_ids=[language],
        language_tag={"my": "Burmese", "lo": "Lao"}[language],
        train_sources=[{"name": source_name, "manifest": str(Path(manifest).resolve()),
                        "sampling_weight": 1.0, "loss_weight": None}],
        validation_sources=[], validation_fraction=0.0,
        seed=seed, mode="full", strategy="fsdp", expected_world_size=1,
        gradient_accumulation_steps=3, expected_train_records=report["accepted_rows"],
        expected_steps_per_epoch=report["accepted_rows"], max_steps=steps, scheduler_steps=steps,
        learning_rate=2e-6 if stage == "S" else 1e-6, warmup_ratio=0.03,
        stop_loss_weight=0.1, resume_from=None, init_adapter_from=None,
        core_model_dir=None, stage_parent_checkpoint=None,
        data_order_resume_version="epoch_seeded_v2", optimizer_foreach=False,
        save_at_steps=sorted({min(200, steps), steps}),
        health_check_step=min(200, steps), health_loss_check_step=0)
    if s_core is not None:
        s_core = Path(s_core).resolve()
        metadata = json.loads((s_core / "export_metadata.json").read_text())
        if metadata.get("status") != "passed" or metadata.get("checkpoint_type") != "full_fsdp_export":
            raise ValueError("S initialization must be a completed native model-only export")
        for filename in ("model.safetensors", "config.json"):
            if not (s_core / filename).is_file():
                raise FileNotFoundError(s_core / filename)
        config.update(core_model_dir=str(s_core), stage_parent_checkpoint=metadata["source_checkpoint"])
    return config


def check_config(config, code_dir=None):
    sources = config["train_sources"]
    if len(sources) != 1 or sources[0].get("loss_weight") is not None:
        raise ValueError("use exactly one prepared manifest with source loss_weight=null")
    report = verify_manifest(sources[0]["manifest"])
    if config.get("resume_from") or config.get("init_adapter_from"):
        raise ValueError("a fresh S/R stage must reset optimizer, scheduler and training offsets")
    if config.get("expected_world_size") != 1 or config.get("gradient_accumulation_steps") != 3:
        raise ValueError("paper preset requires one device and accumulation=3")
    if config.get("stop_loss_weight") != 0.1:
        raise ValueError("paper full loss is flow + 0.1 * stop")
    if report["stage"] == "R" and not config.get("core_model_dir"):
        raise ValueError("R must initialize from the S model export")
    if report["stage"] == "S" and config.get("core_model_dir"):
        raise ValueError("S must initialize from Base")
    result = {"status": "passed", "stage": report["stage"], "records": report["accepted_rows"],
              "weights": report["weights"], "optimizer_reset": True}
    if code_dir:
        sys.path.insert(0, str(Path(code_dir).resolve()))
        from fireredtts3.training.trainer import TrainerConfig, _build_datasets
        native = TrainerConfig(**config)
        native.validate()
        train, _ = _build_datasets(native)
        if len(train) != report["accepted_rows"] or train.dropped_unpaired:
            raise ValueError("native loader changed the prepared target/reference pool")
        expected = {row_id(row): fields(row)["pseudo_label_weight"] for row in read_rows(sources[0]["manifest"])}
        observed = []
        for i in range(len(train)):
            target = train[i]["target"]
            if target.record_id not in expected or target.loss_weight != expected[target.record_id]:
                raise ValueError("native loader changed an individual example weight")
            observed.append(target.loss_weight)
        if not math.isclose(sum(observed) / len(observed), report["weights"]["mean"], abs_tol=1e-12):
            raise ValueError("native loader did not preserve sample weights")
        import torch
        result.update(native_records=len(train), native_weight_mean=sum(observed) / len(observed),
                      native_weight_rows_verified=len(observed), cuda_initialized=torch.cuda.is_initialized())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("configure")
    prepare.add_argument("--template", help="Optional override of the included complete stage template")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--pretrained", required=True)
    prepare.add_argument("--s-core")
    prepare.add_argument("--language", choices=list(RECIPES), default="my")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--config-out", required=True)
    check = sub.add_parser("check")
    check.add_argument("--config", required=True)
    check.add_argument("--code-dir", help="Optional override of the included backbone source")
    run = sub.add_parser("train")
    run.add_argument("--config", required=True)
    run.add_argument("--resume", help="Resume the same stage, not an S to R transition")
    export = sub.add_parser("export")
    export.add_argument("--config", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--checkpoint", help="Defaults to the final configured step")
    args = parser.parse_args(argv)
    if args.command == "configure":
        stage = verify_manifest(args.manifest)["stage"]
        template = Path(args.template) if args.template else Path(__file__).resolve().parent / "configs" / "firered" / f"{args.language}_{stage}.json"
        config = stage_config(json.loads(template.read_text()), args.manifest,
            args.output_dir, args.pretrained, language=args.language, seed=args.seed, s_core=args.s_core)
        external_output(args.output_dir)
        dest = external_output(args.config_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2)
        print(str(dest))
        return
    config = json.loads(Path(args.config).read_text())
    included = use_backbone("firered")
    if args.command == "export":
        from fireredtts3.training.export import main as export_main
        checkpoint = args.checkpoint or str(Path(config["output_dir"]) / f'checkpoint-{config["max_steps"]:08d}')
        return export_main(["--config", str(Path(args.config).resolve()), "--checkpoint", checkpoint,
                           "--expected-step", str(config["max_steps"]),
                           "--output-dir", str(external_output(args.output_dir))])
    if args.command == "check":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print(json.dumps(check_config(config, args.code_dir or included), indent=2))
        return
    external_output(config["output_dir"])
    check_config(config, included)
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("a fresh stage requires an empty output directory")
    from fireredtts3.training.trainer import main as native_main
    arguments = ["--config", str(Path(args.config).resolve())]
    if args.resume:
        arguments += ["--resume-from", args.resume]
    native_main(arguments)


if __name__ == "__main__":
    main()
