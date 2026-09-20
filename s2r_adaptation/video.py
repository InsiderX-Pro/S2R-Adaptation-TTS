"""Convert accepted video-pipeline clips to distinct same-speaker TTS pairs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from .agreement import normalize_for_scoring
from .manifests import read_rows, sha256


def prepare_pairs(input_manifest, output_dir, *, language="my", seed=42,
                  min_target_seconds=.4, max_target_seconds=20,
                  min_prompt_seconds=1, max_prompt_seconds=10):
    if not (0 < min_target_seconds <= max_target_seconds and
            0 < min_prompt_seconds <= max_prompt_seconds):
        raise ValueError("invalid target/reference duration bounds")
    input_manifest, output_dir = Path(input_manifest), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    source_hash = sha256(input_manifest)
    rows, seen, rejected = [], set(), []
    groups = defaultdict(list)
    for source in read_rows(input_manifest):
        key = source.get("item_id")
        if not isinstance(key, str) or not key or key in seen:
            raise ValueError("video rows require unique, nonempty item_id strings")
        seen.add(key)
        reason = None
        text = source.get("final_text")
        duration = float(source.get("duration_ms", 0)) / 1000
        speaker, recording = source.get("speaker_id"), source.get("video_id")
        if source.get("accepted") is not True:
            reason = "not_accepted_by_video_pipeline"
        elif source.get("speaker_pure") is False or not isinstance(speaker, str) or speaker.lower() in {"", "unknown", "overlap"}:
            reason = "invalid_local_speaker"
        elif not isinstance(recording, str) or not recording:
            raise ValueError(f"{key}: video_id is required to scope local speaker labels")
        elif not isinstance(text, str) or not normalize_for_scoring(text):
            reason = "empty_primary_text"
        elif not math.isfinite(duration) or not min_target_seconds <= duration <= max_target_seconds:
            reason = "target_duration_out_of_range"
        elif not source.get("audio_path") or not source.get("audio_sha256"):
            raise ValueError(f"{key}: final audio_path and audio_sha256 are required")
        if reason:
            rejected.append({"id": key, "reason": reason})
            continue
        record = {"id": key, "text": text, "spoken_text": text,
            "audio_path": source["audio_path"], "audio_sha256": source["audio_sha256"],
            "duration": duration, "language_id": language, "video_id": recording,
            "speaker_id": json.dumps([recording, speaker], ensure_ascii=False, separators=(",", ":")),
            "local_speaker_id": speaker, "text_source": "video_pipeline_final_primary_asr"}
        rows.append(record)
        if min_prompt_seconds <= duration <= max_prompt_seconds:
            groups[(recording, speaker)].append(record)
    for members in groups.values():
        members.sort(key=lambda row: row["id"])
    paired = []
    for row in rows:
        candidates = [p for p in groups[(row["video_id"], row["local_speaker_id"]) ]
                      if p["id"] != row["id"] and p["audio_path"] != row["audio_path"]
                      and p["audio_sha256"] != row["audio_sha256"]]
        if not candidates:
            rejected.append({"id": row["id"], "reason": "no_distinct_same_recording_speaker_reference"})
            continue
        # Stable under reordering of the input and independent of Python's hash seed.
        selection = int.from_bytes(hashlib.sha256(f"{seed}:{row['id']}".encode()).digest(), "big")
        prompt = candidates[selection % len(candidates)]
        row.update(prompt_audio_path=prompt["audio_path"], prompt_audio_sha256=prompt["audio_sha256"],
            prompt_text=prompt["text"], prompt_source_id=prompt["id"], prompt_duration=prompt["duration"],
            prompt_language_id=language, prompt_pairing_verified=True,
            prompt_pair_provenance={"protocol": "distinct_same_recording_local_speaker",
                                    "seed": seed, "speaker_identity_status": "automatic_diarization"})
        paired.append(row)
    if not paired:
        raise ValueError("no eligible target/reference pairs")
    if sha256(input_manifest) != source_hash:
        raise ValueError("video input manifest changed during pairing")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    report = {"input_rows": len(seen), "paired_rows": len(paired), "excluded_rows": len(rejected),
              "exclusion_reasons": dict(Counter(row["reason"] for row in rejected)),
              "input_sha256": source_hash, "language": language, "seed": seed,
              "reference_seconds": [min_prompt_seconds, max_prompt_seconds]}
    with tempfile.TemporaryDirectory(prefix=".pairs-", dir=output_dir.parent) as directory:
        staging = Path(directory) / "prepared"
        staging.mkdir()
        for name, records in [("primary.jsonl", paired), ("excluded.jsonl", rejected)]:
            with (staging / name).open("w", encoding="utf-8", newline="\n") as stream:
                for row in records:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["primary_sha256"] = sha256(staging / "primary.jsonl")
        (staging / "pairing_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output_dir)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="my")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_pairs(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
