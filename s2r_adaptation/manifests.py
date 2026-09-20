"""Join frozen ASR evidence to primary-text training pairs; keep pairs intact."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import os

from .agreement import disagreement, normalize_for_scoring, reliability_weight


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def row_id(row):
    candidates = [row[k] for k in ("id", "target_id") if k in row]
    if not candidates or not isinstance(candidates[0], str) or not candidates[0]:
        raise ValueError("every row needs a nonempty string id or target_id")
    if any(value != candidates[0] for value in candidates):
        raise ValueError("id and target_id disagree")
    return candidates[0]


def fields(row):
    label = row.get("label", row)
    if not isinstance(label, dict):
        raise ValueError("label must be an object")
    return label


def checked_evidence(row, target, asr_format, require_audio_sha):
    """An empty successful hypothesis is valid; absent/failed requests are errors."""
    status = row.get("status")
    if status is not None and status not in ("success", "passed", "complete", "completed"):
        raise ValueError("secondary_asr_failed")
    if row.get("error"):
        raise ValueError("secondary_asr_error")
    if status is None and asr_format != "archived":
        raise ValueError("secondary_asr_status_missing")
    if status is None and not isinstance(row.get("canonical"), dict):
        raise ValueError("archived_asr_requires_canonical_evidence")
    hypothesis = row.get("hypothesis", row.get("text"))
    if not isinstance(hypothesis, str):
        raise ValueError("secondary_transcript_missing")
    primary = fields(target)["text"]
    if "reference" in row and row["reference"] != primary:
        raise ValueError("primary_transcript_binding_mismatch")
    left, right = target.get("audio_sha256"), row.get("audio_sha256")
    if require_audio_sha and (not left or not right):
        raise ValueError("audio_sha256_required")
    if left and right and left != right:
        raise ValueError("audio_sha256_mismatch")
    d = disagreement(primary, hypothesis)
    canonical = row.get("canonical")
    if canonical is not None:
        if not isinstance(canonical, dict) or "cer" not in canonical:
            raise ValueError("invalid_canonical_evidence")
        archived_d = float(canonical["cer"])
        if not math.isfinite(archived_d) or not math.isclose(d, archived_d, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("archived_cer_disagrees_with_recomputed_transcripts")
    return d


def build_manifest(primary, output_dir, *, secondary=None, stage="R", gamma=3.0,
                   w_min=0.10, asr_format="status", on_asr_error="fail",
                   require_audio_sha=False):
    reliability_weight(0, gamma, w_min)
    if stage not in ("S", "R") or on_asr_error not in ("fail", "exclude"):
        raise ValueError("invalid stage or ASR failure policy")
    if asr_format not in ("status", "archived"):
        raise ValueError("invalid ASR format")
    if (stage == "R") != (secondary is not None):
        raise ValueError("R requires secondary ASR; S takes original synthetic pairs only")
    primary, output_dir = Path(primary), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    secondary_index = {}
    if secondary is not None:
        for row in read_rows(secondary):
            key = row_id(row)
            if key in secondary_index:
                raise ValueError(f"duplicate secondary ASR id: {key}")
            secondary_index[key] = row
    source_hashes = {"primary": sha256(primary)}
    if secondary is not None:
        source_hashes["secondary"] = sha256(secondary)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    seen, weights, reasons, observers = set(), [], Counter(), Counter()
    # Nothing is published if validation fails midway through a corpus.
    with tempfile.TemporaryDirectory(prefix=".s2r-", dir=output_dir.parent) as scratch:
        staging = Path(scratch) / "prepared"
        staging.mkdir()
        with (staging / "train.jsonl").open("w", encoding="utf-8", newline="\n") as accepted, \
             (staging / "excluded.jsonl").open("w", encoding="utf-8", newline="\n") as excluded:
            for row in read_rows(primary):
                key = row_id(row)
                if key in seen:
                    raise ValueError(f"duplicate primary id: {key}")
                seen.add(key)
                label = fields(row)
                primary_text = label.get("text")
                if not isinstance(primary_text, str):
                    raise ValueError(f"{key}: primary transcript must be a string")
                reason = None
                if not normalize_for_scoring(primary_text):
                    reason = "empty_normalized_primary"
                elif stage == "R":
                    evidence = secondary_index.get(key)
                    try:
                        if evidence is None:
                            raise ValueError("secondary_asr_missing")
                        d = checked_evidence(evidence, row, asr_format, require_audio_sha)
                    except ValueError as exc:
                        if on_asr_error == "fail":
                            raise ValueError(f"{key}: {exc}") from exc
                        reason = str(exc)
                if reason:
                    reasons[reason] += 1
                    excluded.write(json.dumps({"id": key, "reason": reason}) + "\n")
                    continue
                weight = 1.0 if stage == "S" else reliability_weight(d, gamma, w_min)
                label["pseudo_label_weight"] = weight
                # Legacy readers may prefer loss_weight; prevent stale values.
                if "loss_weight" in label:
                    label["loss_weight"] = weight
                row["reliability"] = {"stage": stage, "weight": weight,
                    "normalization": "NFC_drop_whitespace_PSC_codepoints",
                    "primary_text_unchanged": True}
                if stage == "R":
                    observer = evidence.get("model_id", "unspecified")
                    observers[str(observer)] += 1
                    row["reliability"].update(disagreement=d, gamma=gamma, w_min=w_min,
                                               secondary_model=observer)
                accepted.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                weights.append(weight)
        if not weights:
            raise ValueError("no eligible training examples")
        if source_hashes["primary"] != sha256(primary) or (
            secondary is not None and source_hashes["secondary"] != sha256(secondary)
        ):
            raise ValueError("input changed during preparation")
        report = {"stage": stage, "input_rows": len(seen), "accepted_rows": len(weights),
            "excluded_rows": sum(reasons.values()), "exclusion_reasons": dict(reasons),
            "unused_secondary_rows": len(set(secondary_index) - seen),
            "gamma": gamma if stage == "R" else None, "w_min": w_min if stage == "R" else None,
            "weights": {"mean": statistics.fmean(weights), "population_std": statistics.pstdev(weights),
                        "min": min(weights), "max": max(weights),
                        "at_floor": sum(w == w_min for w in weights) if stage == "R" else 0,
                        "at_one": sum(w == 1 for w in weights)},
            "secondary_models": dict(observers), "input_sha256": source_hashes,
            "train_sha256": sha256(staging / "train.jsonl"),
            "formula": "max(w_min, (1 - min(d, 1)) ** gamma)",
            "loss_reduction": "sum(weight_i * full_sample_loss_i) / number_of_examples",
            "asr_format": asr_format, "on_asr_error": on_asr_error}
        (staging / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", required=True, help="JSONL training pairs with original ASR1 text")
    parser.add_argument("--secondary", help="JSONL ASR2 results joined by exact id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=["S", "R"], default="R")
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--w-min", type=float, default=0.10)
    parser.add_argument("--asr-format", choices=["status", "archived"], default="status")
    parser.add_argument("--on-asr-error", choices=["fail", "exclude"], default="fail")
    parser.add_argument("--require-audio-sha", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(build_manifest(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
