"""Manifest loading and deterministic prompt/target pairing for TTS training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler


_AUDIO_FIELDS = ("audio_path", "wav_path", "audio", "path")
_TEXT_FIELDS = ("spoken_text", "text", "final_text", "transcript", "sentence")
# Synthetic manifests identify the fixed TTS speaker with ``voice`` while the
# real-data manifests use diarization-local speaker IDs.  Treating ``voice`` as
# the final alias keeps same-speaker prompt pairing valid for both contracts.
_SPEAKER_FIELDS = ("local_speaker_id", "speaker_id", "speaker", "spk_id", "voice")
_ID_FIELDS = ("id", "utt_id", "audio_id", "key")


def _first_nonempty(row: Mapping[str, Any], fields: Sequence[str], default: Any = None) -> Any:
    for field in fields:
        value = row.get(field)
        if value is not None and value != "":
            return value
    return default


def _stable_u64(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


@dataclass(frozen=True)
class SpeechRecord:
    """Normalized record used by the FireRed trainer."""

    record_id: str
    audio_path: str
    text: str
    speaker_id: str
    language_id: str = "my"
    duration: float | None = None
    split: str = "train"
    source_name: str = "default"
    loss_weight: float = 1.0
    sampling_mass: float = 1.0
    video_id: str | None = None
    audio_sha256: str | None = None
    prompt_audio_path: str | None = None
    prompt_text: str | None = None
    prompt_language_id: str | None = None
    raw: Mapping[str, Any] | None = None

    @property
    def speaker_key(self) -> str:
        # Prefixing prevents unrelated source-local speaker IDs from colliding.
        return f"{self.source_name}:{self.speaker_id}"


@dataclass(frozen=True)
class SourceSpec:
    """One input corpus in a real/synthetic mixture.

    ``sampling_weight`` is corpus-level probability mass, not per-record mass.
    ``loss_weight`` overrides a manifest's pseudo-label weight when non-null.
    """

    manifest: str
    name: str
    sampling_weight: float = 1.0
    loss_weight: float | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceSpec":
        return cls(
            manifest=str(value["manifest"]),
            name=str(value.get("name") or Path(str(value["manifest"])).stem),
            sampling_weight=float(value.get("sampling_weight", 1.0)),
            loss_weight=(None if value.get("loss_weight") is None else float(value["loss_weight"])),
        )


def _iter_jsonl_paths(manifest: str | Path) -> Iterator[Path]:
    """Yield JSONL files from a JSONL manifest or OmniVoice ``data.lst``."""

    path = Path(manifest)
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    if path.suffix.lower() in {".jsonl", ".json"}:
        yield path
        return

    # OmniVoice data.lst rows are: audio_tar jsonl row_count seconds.
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_no, line in enumerate(stream, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_no}: expected at least two fields")
            label_path = Path(fields[1])
            if not label_path.is_absolute():
                label_path = path.parent / label_path
            if not label_path.is_file():
                raise FileNotFoundError(f"label manifest not found: {label_path}")
            yield label_path


def _row_to_record(
    row: Mapping[str, Any],
    *,
    source: SourceSpec,
    fallback_id: str,
) -> SpeechRecord:
    audio_path = str(_first_nonempty(row, _AUDIO_FIELDS, "")).strip()
    text = str(_first_nonempty(row, _TEXT_FIELDS, "")).strip()
    record_id = str(_first_nonempty(row, _ID_FIELDS, fallback_id))
    speaker_id = str(_first_nonempty(row, _SPEAKER_FIELDS, "")).strip()
    if not audio_path:
        raise ValueError(f"{fallback_id}: missing audio path")
    if not text:
        raise ValueError(f"{fallback_id}: missing text")
    if not speaker_id:
        raise ValueError(f"{fallback_id}: missing speaker id; same-speaker prompts are required")

    duration_value = _first_nonempty(
        row,
        ("audio_duration", "duration", "duration_sec", "alignment_audio_duration_sec"),
    )
    duration = None if duration_value is None else float(duration_value)
    manifest_weight = float(row.get("pseudo_label_weight", row.get("loss_weight", 1.0)))
    # server180 experiment contract: never silently replace an explicit
    # per-record reliability weight with a corpus-wide constant.
    explicit_weight = "pseudo_label_weight" in row or "loss_weight" in row
    if source.loss_weight is not None and explicit_weight and not math.isclose(
        float(source.loss_weight), manifest_weight, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(f"{fallback_id}: source loss_weight conflicts with explicit sample weight")
    loss_weight = source.loss_weight if source.loss_weight is not None else manifest_weight
    if not math.isfinite(float(loss_weight)) or not 0.0 < float(loss_weight) <= 1.0:
        raise ValueError(f"{fallback_id}: experiment sample weight must be finite in (0, 1]")
    return SpeechRecord(
        record_id=record_id,
        audio_path=audio_path,
        text=text,
        speaker_id=speaker_id,
        language_id=str(row.get("language_id", row.get("lang", "my"))),
        duration=duration,
        split=str(row.get("split", "train")),
        source_name=source.name,
        loss_weight=float(loss_weight),
        sampling_mass=float(source.sampling_weight),
        video_id=(None if row.get("video_id") is None else str(row["video_id"])),
        audio_sha256=(None if row.get("audio_sha256") is None else str(row["audio_sha256"])),
        prompt_audio_path=(
            None if row.get("prompt_audio_path") is None else str(row["prompt_audio_path"])
        ),
        prompt_text=(None if row.get("prompt_text") is None else str(row["prompt_text"])),
        prompt_language_id=(
            None
            if row.get("prompt_language_id") is None
            else str(row["prompt_language_id"])
        ),
        raw=dict(row),
    )


def load_source_records(
    source: SourceSpec,
    *,
    language_ids: set[str] | None = None,
    min_duration: float = 0.4,
    max_duration: float = 20.0,
    required_split: str | None = None,
    check_audio_exists: bool = True,
) -> list[SpeechRecord]:
    """Load and normalize one source, applying only explicit hard filters."""

    records: list[SpeechRecord] = []
    seen_ids: set[str] = set()
    for jsonl_path in _iter_jsonl_paths(source.manifest):
        with jsonl_path.open("r", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                fallback_id = f"{jsonl_path}:{line_no}"
                record = _row_to_record(row, source=source, fallback_id=fallback_id)
                if language_ids and record.language_id not in language_ids:
                    continue
                if required_split and record.split != required_split:
                    continue
                if record.duration is not None and not (min_duration <= record.duration <= max_duration):
                    continue
                if check_audio_exists and not Path(record.audio_path).is_file():
                    raise FileNotFoundError(f"audio not found for {record.record_id}: {record.audio_path}")
                unique_id = f"{source.name}:{record.record_id}"
                if unique_id in seen_ids:
                    raise ValueError(f"duplicate record id in {source.name}: {record.record_id}")
                seen_ids.add(unique_id)
                records.append(record)
    if not records:
        raise ValueError(f"source {source.name!r} produced no eligible records")
    per_record_mass = source.sampling_weight / len(records)
    return [replace(item, sampling_mass=per_record_mass) for item in records]


def load_mixture_records(
    sources: Sequence[SourceSpec],
    **kwargs: Any,
) -> list[SpeechRecord]:
    """Load sources and turn corpus probability mass into per-record mass."""

    all_records: list[SpeechRecord] = []
    for source in sources:
        if source.sampling_weight <= 0:
            continue
        records = load_source_records(source, **kwargs)
        per_record_mass = source.sampling_weight / len(records)
        all_records.extend(replace(item, sampling_mass=per_record_mass) for item in records)
    if not all_records:
        raise ValueError("no positive-weight sources")
    return all_records


def split_records_by_group(
    records: Sequence[SpeechRecord],
    validation_fraction: float,
    *,
    seed: int,
    group_field: str = "video_id",
) -> tuple[list[SpeechRecord], list[SpeechRecord]]:
    """Deterministically split whole groups to prevent source leakage."""

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0:
        return list(records), []

    threshold = int(validation_fraction * (1 << 64))
    train: list[SpeechRecord] = []
    valid: list[SpeechRecord] = []
    for item in records:
        group = getattr(item, group_field, None) or item.speaker_key
        key = f"split:{seed}:{item.source_name}:{group}"
        (valid if _stable_u64(key) < threshold else train).append(item)
    if not train or not valid:
        raise ValueError(
            f"group split yielded train={len(train)} valid={len(valid)}; "
            "adjust validation_fraction or provide an explicit validation manifest"
        )
    return train, valid


class PromptPairedDataset(Dataset[dict[str, Any]]):
    """Pair every target with a distinct, same-speaker prompt utterance."""

    def __init__(
        self,
        records: Sequence[SpeechRecord],
        *,
        seed: int = 42,
        prompt_min_duration: float = 1.0,
        prompt_max_duration: float = 10.0,
        drop_unpaired: bool = True,
    ):
        self.seed = int(seed)
        self.epoch = 0
        source_masses: dict[str, float] = {}
        by_speaker: dict[str, list[SpeechRecord]] = {}
        for item in records:
            source_masses[item.source_name] = source_masses.get(item.source_name, 0.0) + item.sampling_mass
            if item.duration is None or prompt_min_duration <= item.duration <= prompt_max_duration:
                by_speaker.setdefault(item.speaker_key, []).append(item)

        self._prompt_pool = by_speaker
        self.records: list[SpeechRecord] = []
        self.dropped_unpaired = 0
        for item in records:
            explicit = bool(item.prompt_audio_path)
            candidates = by_speaker.get(item.speaker_key, [])
            has_distinct = any(candidate.record_id != item.record_id for candidate in candidates)
            if explicit or has_distinct:
                self.records.append(item)
            elif drop_unpaired:
                self.dropped_unpaired += 1
            else:
                raise ValueError(f"no distinct same-speaker prompt for {item.record_id}")
        if not self.records:
            raise ValueError("no prompt-pairable records")
        # Pair filtering can remove different fractions from real and synthetic
        # corpora.  Re-normalize within each source so requested corpus-level
        # sampling mass remains exact after those drops.
        eligible_counts: dict[str, int] = {}
        for item in self.records:
            eligible_counts[item.source_name] = eligible_counts.get(item.source_name, 0) + 1
        self.records = [
            replace(
                item,
                sampling_mass=source_masses[item.source_name] / eligible_counts[item.source_name],
            )
            for item in self.records
        ]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _choose_prompt(self, target: SpeechRecord) -> SpeechRecord:
        candidates = [
            item
            for item in self._prompt_pool[target.speaker_key]
            if item.record_id != target.record_id
        ]
        if not candidates:
            raise RuntimeError(f"no prompt candidate for {target.record_id}")
        index = _stable_u64(f"prompt:{self.seed}:{self.epoch}:{target.source_name}:{target.record_id}")
        return candidates[index % len(candidates)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        target = self.records[index]
        if target.prompt_audio_path:
            prompt = SpeechRecord(
                record_id=f"{target.record_id}:explicit_prompt",
                audio_path=target.prompt_audio_path,
                text=target.prompt_text or "",
                speaker_id=target.speaker_id,
                language_id=target.prompt_language_id or target.language_id,
                source_name=target.source_name,
                audio_sha256=(target.raw or {}).get("prompt_audio_sha256"),
            )
        else:
            prompt = self._choose_prompt(target)
        return {"target": target, "prompt": prompt}


class DistributedWeightedSampler(Sampler[int]):
    """Deterministic distributed multinomial sampler for corpus mixtures."""

    def __init__(
        self,
        weights: Sequence[float],
        *,
        num_replicas: int = 1,
        rank: int = 0,
        samples_per_epoch: int | None = None,
        seed: int = 42,
    ):
        if not weights or any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("invalid distributed rank")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        requested = len(weights) if samples_per_epoch is None else int(samples_per_epoch)
        self.num_samples = int(math.ceil(requested / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterable[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        global_indices = torch.multinomial(
            self.weights,
            self.total_size,
            replacement=True,
            generator=generator,
        ).tolist()
        return iter(global_indices[self.rank : self.total_size : self.num_replicas])


def dataset_summary(dataset: PromptPairedDataset) -> dict[str, Any]:
    by_source: dict[str, int] = {}
    hours_by_source: dict[str, float] = {}
    for item in dataset.records:
        by_source[item.source_name] = by_source.get(item.source_name, 0) + 1
        hours_by_source[item.source_name] = hours_by_source.get(item.source_name, 0.0) + (
            item.duration or 0.0
        ) / 3600.0
    return {
        "eligible_records": len(dataset),
        "dropped_unpaired": dataset.dropped_unpaired,
        "records_by_source": by_source,
        "hours_by_source": hours_by_source,
    }
