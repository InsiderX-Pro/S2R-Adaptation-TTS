"""Encode explicit target/reference audio pairs with OmniVoice's Higgs codec."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from .backbones import external_output
from .firered import verify_manifest
from .manifests import fields, read_rows, row_id, sha256


class HiggsEncoder:
    def __init__(self, model_dir, device="cpu"):
        import torch
        from transformers import AutoFeatureExtractor, HiggsAudioV2TokenizerModel
        model_dir = Path(model_dir).resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError("Provide the downloaded, pinned audio_tokenizer directory")
        self.extractor = AutoFeatureExtractor.from_pretrained(model_dir, local_files_only=True)
        self.model = HiggsAudioV2TokenizerModel.from_pretrained(
            model_dir, local_files_only=True).to(device).eval()
        self.sample_rate = self.extractor.sampling_rate
        self.device = torch.device(device)

    @staticmethod
    def duration(path):
        import soundfile as sf
        return sf.info(path).duration

    def __call__(self, path):
        import torch
        import torchaudio
        import soundfile as sf
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio).mean(dim=1)
        if not wave.numel() or not torch.isfinite(wave).all():
            raise ValueError("empty or non-finite audio")
        if rate != self.sample_rate:
            wave = torchaudio.functional.resample(wave, rate, self.sample_rate)
        inputs = self.extractor(raw_audio=wave.numpy(), sampling_rate=self.sample_rate,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self.model.encode(inputs["input_values"]).audio_codes.squeeze(0).cpu()


def prepare_codec(manifest, output_dir, encoder, *, codec_identity, channels=8, vocab_size=1024):
    import torch
    manifest = Path(manifest).resolve()
    report = verify_manifest(manifest)
    output = external_output(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".codec-", dir=output.parent))
    cache = {}
    count = 0

    def encode(path, claimed_digest, minimum, maximum):
        path = Path(path).expanduser()
        if not path.is_absolute():
            path = manifest.parent / path
        path = path.resolve()
        if hasattr(encoder, "duration") and not minimum <= encoder.duration(path) <= maximum:
            raise ValueError(f"audio duration must be between {minimum} and {maximum} seconds")
        digest = sha256(path)
        if claimed_digest and claimed_digest != digest:
            raise ValueError("audio bytes differ from the manifest digest")
        if digest not in cache:
            value = encoder(path)
            if value.ndim != 2 or value.shape[0] != channels or value.shape[1] == 0:
                raise ValueError("codec output must have shape [codebooks, frames]")
            if value.is_floating_point() or (value < 0).any() or (value >= vocab_size).any():
                raise ValueError("invalid acoustic codec IDs")
            relative = "tokens/" + digest + ".pt"
            token_path = temporary / relative
            token_path.parent.mkdir(exist_ok=True)
            torch.save(value.to(dtype=torch.int16, device="cpu").contiguous(), token_path)
            cache[digest] = (relative, value.shape[1], sha256(token_path))
        return digest, cache[digest]

    try:
        with (temporary / "pairs.jsonl").open("w", encoding="utf-8") as stream:
            for row in read_rows(manifest):
                label = fields(row)
                required = ("audio_path", "prompt_audio_path", "text", "prompt_text")
                if any(not isinstance(label.get(k), str) or not label[k] for k in required):
                    raise ValueError(f"{row_id(row)} needs explicit target/reference audio and text")
                target_sha, target = encode(label["audio_path"], label.get("audio_sha256"), 0.4, 20)
                prompt_sha, prompt = encode(label["prompt_audio_path"], label.get("prompt_audio_sha256"), 1, 10)
                if target_sha == prompt_sha:
                    raise ValueError("target and reference must be different audio")
                record = {"id": row_id(row), "text": label["text"], "prompt_text": label["prompt_text"],
                    "language_id": label.get("language_id", "my"),
                    "target_tokens": target[0], "prompt_tokens": prompt[0],
                    "target_frames": target[1], "prompt_frames": prompt[1],
                    "target_tokens_sha256": target[2], "prompt_tokens_sha256": prompt[2],
                    "audio_sha256": target_sha, "prompt_audio_sha256": prompt_sha,
                    "pseudo_label_weight": label["pseudo_label_weight"]}
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        metadata = {"schema_version": 1, "stage": report["stage"], "records": count,
                    "source_manifest_sha256": sha256(manifest), "codec": codec_identity,
                    "num_channels": channels, "vocab_size": vocab_size,
                    "index_sha256": sha256(temporary / "pairs.jsonl")}
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
        temporary.rename(output)
    except BaseException:
        shutil.rmtree(temporary)
        raise
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codec-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    model_dir = Path(args.codec_dir).resolve()
    identity = {"type": "HiggsAudioV2TokenizerModel", "config_sha256": sha256(model_dir / "config.json"),
                "weights": {p.name: sha256(p) for p in sorted(model_dir.glob("*.safetensors"))}}
    if not identity["weights"]:
        raise FileNotFoundError("codec .safetensors weights are required")
    print(json.dumps(prepare_codec(args.manifest, args.output_dir,
        HiggsEncoder(model_dir, args.device), codec_identity=identity), indent=2))


if __name__ == "__main__":
    main()
