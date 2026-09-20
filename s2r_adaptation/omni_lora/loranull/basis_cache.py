"""Versioned, hashable safetensors cache for per-module null bases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import save_file


SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_basis(name: str, basis: torch.Tensor) -> torch.Tensor:
    if basis.ndim != 2 or min(basis.shape) <= 0:
        raise ValueError(f"basis {name!r} must have shape [in_features, rank]")
    value = basis.detach().to(device="cpu", dtype=torch.float32).contiguous()
    gram = value.mT @ value
    identity = torch.eye(value.shape[1], dtype=value.dtype)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError(f"basis {name!r} is not orthonormal")
    return value


def save_basis_cache(
    path: str | Path,
    bases: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
) -> str:
    """Atomically-ish write a self-describing basis cache and return SHA-256."""

    destination = Path(path)
    if destination.suffix != ".safetensors":
        raise ValueError("basis cache path must end in .safetensors")
    if not bases:
        raise ValueError("cannot save an empty basis cache")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {name: _validate_basis(name, value) for name, value in sorted(bases.items())}
    payload = dict(metadata)
    payload["schema_version"] = SCHEMA_VERSION
    payload["modules"] = {
        name: {"in_features": value.shape[0], "rank": value.shape[1]}
        for name, value in tensors.items()
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, temporary, metadata={"loranull_metadata": encoded})
    temporary.replace(destination)
    sidecar = destination.with_suffix(destination.suffix + ".json")
    sidecar_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    sidecar_temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    sidecar_temporary.replace(sidecar)
    return sha256_file(destination)


def load_basis_cache(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"null-basis cache not found: {source}")
    observed = sha256_file(source)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            f"null-basis SHA-256 mismatch: expected {expected_sha256}, got {observed}"
        )
    with safe_open(source, framework="pt", device="cpu") as handle:
        header = handle.metadata() or {}
        if "loranull_metadata" not in header:
            raise RuntimeError("basis cache lacks lorannull metadata")
        metadata = json.loads(header["loranull_metadata"])
        bases = {name: _validate_basis(name, handle.get_tensor(name)) for name in handle.keys()}
    sidecar = source.with_suffix(source.suffix + ".json")
    if not sidecar.is_file():
        raise RuntimeError(f"basis cache sidecar not found: {sidecar}")
    sidecar_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if sidecar_metadata != metadata:
        raise RuntimeError("basis-cache sidecar disagrees with safetensors metadata")
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"unsupported basis-cache schema: {metadata.get('schema_version')}")
    declared = metadata.get("modules", {})
    if set(declared) != set(bases):
        raise RuntimeError("basis-cache tensor keys disagree with metadata")
    for name, basis in bases.items():
        shape = declared[name]
        if [shape.get("in_features"), shape.get("rank")] != list(basis.shape):
            raise RuntimeError(f"basis shape metadata mismatch for {name}")
    metadata["cache_sha256"] = observed
    return bases, metadata
