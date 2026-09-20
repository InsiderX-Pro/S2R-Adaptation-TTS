"""Paired reference conditioning and target-only masked acoustic supervision."""
import json
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset

from .firered import verify_manifest
from .manifests import fields, read_rows, row_id, sha256


class PairedCodecDataset(Dataset):
    def __init__(self, directory, manifest):
        self.directory = Path(directory).resolve()
        self.metadata = json.loads((self.directory / "metadata.json").read_text())
        report = verify_manifest(manifest)
        index = self.directory / "pairs.jsonl"
        if self.metadata["source_manifest_sha256"] != sha256(manifest) or \
                self.metadata["stage"] != report["stage"] or self.metadata["index_sha256"] != sha256(index):
            raise ValueError("codec preparation does not match this manifest/stage")
        self.rows = list(read_rows(index))
        primary = {row_id(row): fields(row) for row in read_rows(manifest)}
        if len(self.rows) != len(primary) or len(self.rows) != self.metadata["records"]:
            raise ValueError("codec row count differs from prepared data")
        seen = set()
        for row in self.rows:
            identity = row_id(row)
            if identity in seen or identity not in primary:
                raise ValueError("duplicate/unknown codec ID")
            seen.add(identity)
            original = primary[identity]
            for key in ("text", "prompt_text", "pseudo_label_weight"):
                if row[key] != original[key]:
                    raise ValueError("codec text/reference/weight differs from prepared data")
            for key in ("audio_sha256", "prompt_audio_sha256"):
                if original.get(key) and row[key] != original[key]:
                    raise ValueError("codec audio identity differs from prepared data")
        self._verified = set()

    def __len__(self):
        return len(self.rows)

    def _load(self, relative, digest):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory):
            raise ValueError("codec token path escapes prepared directory")
        if path not in self._verified:
            if sha256(path) != digest:
                raise ValueError("codec token checksum mismatch")
            self._verified.add(path)
        tokens = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2 or \
                tokens.shape[0] != self.metadata["num_channels"] or tokens.shape[1] == 0 or \
                tokens.is_floating_point() or (tokens < 0).any() or (tokens >= self.metadata["vocab_size"]).any():
            raise ValueError("invalid cached acoustic tokens")
        return tokens.long()

    def __getitem__(self, index):
        row = self.rows[index]
        return {**row, "target": self._load(row["target_tokens"], row["target_tokens_sha256"]),
                "prompt": self._load(row["prompt_tokens"], row["prompt_tokens_sha256"])}


class PairedProcessor:
    def __init__(self, tokenizer, channels, audio_mask_id, *, mask_ratio_range=(0, 1), drop_cond_ratio=0.1):
        self.tokenizer, self.channels, self.mask_id = tokenizer, channels, audio_mask_id
        self.text_tokenizer = tokenizer
        self.mask_range, self.drop_cond = mask_ratio_range, drop_cond_ratio

    def __call__(self, sample):
        target, prompt = sample["target"], sample["prompt"]
        if target.shape[0] != self.channels or prompt.shape[0] != self.channels:
            raise ValueError("codec/model codebook counts differ")
        dropped = random.random() < self.drop_cond
        ratio = random.uniform(*self.mask_range)
        masked = torch.rand(target.shape) < ratio
        noisy = target.clone()
        noisy[masked] = self.mask_id
        target_labels = target.clone()
        target_labels[~masked] = -100
        if dropped:
            inputs, labels = noisy, target_labels
            audio_mask = torch.ones(inputs.shape[1], dtype=torch.bool)
        else:
            style = f'<|lang_start|>{sample["language_id"]}<|lang_end|><|instruct_start|>None<|instruct_end|>'
            # Preserve both transcripts. Scoring normalization never reaches conditioning text.
            text = "<|text_start|>" + sample["prompt_text"] + " " + sample["text"] + "<|text_end|>"
            prefix = torch.cat([self.tokenizer(s, return_tensors="pt").input_ids for s in (style, text)], dim=1)
            prefix = prefix.repeat(self.channels, 1)
            inputs = torch.cat([prefix, prompt, noisy], dim=1)
            labels = torch.cat([torch.full_like(prefix, -100), torch.full_like(prompt, -100), target_labels], dim=1)
            audio_mask = torch.arange(inputs.shape[1]) >= prefix.shape[1]
        return {"input_ids": inputs, "labels": labels, "audio_mask": audio_mask,
                "length": inputs.shape[1], "sample_weight": sample["pseudo_label_weight"]}


def collate_pairs(samples, collator):
    batch = collator(samples)
    batch["sample_weights"] = torch.tensor([s["sample_weight"] for s in samples], dtype=torch.float32)
    return batch
