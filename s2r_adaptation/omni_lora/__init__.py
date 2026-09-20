"""Dependency-free vanilla LoRA and LoRA-Null adapters for OmniVoice.

The module names are intentionally checked against the local OmniVoice/Qwen3
checkout. A changed upstream architecture fails closed instead of silently
training the wrong parameters.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F

from .loranull.basis_cache import load_basis_cache
from .loranull.math import factor_projected_weight, projected_factor_relative_error


ADAPTER_FILE = "adapter_model.safetensors"
PEFT_CONFIG_FILE = "peft_config.json"
AUDIT_FILE = "trainable_audit.json"
COMPLETE_FILE = "COMPLETE.json"
TARGET_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
ROOT_PROJECTIONS = (
    "audio_heads", "audio_embeddings_delta", "audio_heads_delta", "text_embeddings"
)


@dataclass(frozen=True)
class PeftSpec:
    experiment_id: str
    strategy: str
    base_model: str
    rank: int | None = None
    alpha: float | None = None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = TARGET_PROJECTIONS
    layer_indices: tuple[int, ...] = tuple(range(28))
    adapter_layers: int | None = None
    bottleneck: int | None = None
    routing: str = "none"
    initialization: str = "vanilla"
    null_basis_path: str | None = None
    null_basis_sha256: str | None = None
    base_model_sha256: str | None = None
    null_init_scale: float = 1.0
    freeze_a: bool = False
    hard_project_a: bool = False

    def validate(self) -> None:
        if self.strategy == "lora":
            if not self.experiment_id.lower().startswith("p2"):
                raise ValueError("phase-2 LoRA experiment IDs must start with P2")
            if self.rank is None or self.rank <= 0 or self.alpha is None or self.alpha <= 0:
                raise ValueError("LoRA rank and alpha must be positive")
            if not 0 <= self.dropout < 1:
                raise ValueError("LoRA dropout must be in [0, 1)")
            if not self.target_modules or not set(self.target_modules).issubset(
                set(TARGET_PROJECTIONS) | set(MLP_PROJECTIONS) | set(ROOT_PROJECTIONS)
            ):
                raise ValueError(f"unsupported LoRA targets: {self.target_modules}")
            if not self.layer_indices or len(set(self.layer_indices)) != len(self.layer_indices):
                raise ValueError("layer_indices must be non-empty and unique")
            if min(self.layer_indices) < 0 or max(self.layer_indices) >= 28:
                raise ValueError("layer_indices must be within [0, 27]")
            if self.routing not in {
                "none", "prediction", "text_only", "text_prediction", "target_region"
            }:
                raise ValueError(f"unsupported LoRA routing mode: {self.routing}")
            if self.initialization not in {"vanilla", "lora_null"}:
                raise ValueError(f"unsupported LoRA initialization: {self.initialization}")
            if self.initialization == "lora_null":
                if not self.null_basis_path:
                    raise ValueError("LoRA-Null requires null_basis_path")
                if not self.null_basis_sha256 or len(self.null_basis_sha256) != 64:
                    raise ValueError("LoRA-Null requires a 64-character basis SHA-256")
                if not self.base_model_sha256 or len(self.base_model_sha256) != 64:
                    raise ValueError("LoRA-Null requires the base model SHA-256")
                if self.null_init_scale < 0:
                    raise ValueError("null_init_scale must be non-negative")
                if "text_embeddings" in self.target_modules:
                    raise ValueError("LoRA-Null currently supports Linear targets, not embeddings")
                if self.freeze_a and self.hard_project_a:
                    raise ValueError("freeze_a and hard_project_a are mutually exclusive")
            elif self.freeze_a or self.hard_project_a:
                raise ValueError(
                    "freeze_a and hard_project_a are only defined for lora_null initialization"
                )
        elif self.strategy == "residual_adapter":
            if self.experiment_id.lower() != "e5":
                raise ValueError("Residual adapters are reserved for experiment E5")
            if (self.adapter_layers, self.bottleneck) != (4, 64):
                raise ValueError("E5 requires the final 4 blocks and bottleneck=64")
        else:
            raise ValueError(f"Unsupported PEFT strategy: {self.strategy}")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PeftSpec":
        value = dict(value)
        if "target_modules" in value:
            value["target_modules"] = tuple(value["target_modules"])
        if "layer_indices" in value:
            value["layer_indices"] = tuple(int(item) for item in value["layer_indices"])
        spec = cls(**value)
        spec.validate()
        return spec

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["target_modules"] = list(self.target_modules)
        value["layer_indices"] = list(self.layer_indices)
        return value


class LoRALinear(nn.Module):
    """Frozen Linear plus a trainable low-rank delta."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        routing: str = "none",
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.routing = routing
        self.route_mask: torch.Tensor | None = None
        factory = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.lora_A = nn.Linear(base_layer.in_features, rank, bias=False, **factory)
        self.lora_B = nn.Linear(rank, base_layer.out_features, bias=False, **factory)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        if self.scaling == 0.0:
            return base
        delta = self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling
        if self.routing != "none":
            if self.route_mask is None:
                raise RuntimeError(f"LoRA route mask is unset for routing={self.routing}")
            expected = inputs.shape[:-1]
            if tuple(self.route_mask.shape) != tuple(expected):
                raise ValueError(
                    f"LoRA route mask shape {tuple(self.route_mask.shape)} != {tuple(expected)}"
                )
            delta = delta * self.route_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return base + delta


class LoRANullLinear(LoRALinear):
    """Centered LoRA-Null branch that leaves the frozen base weight untouched.

    The forward delta is ``s * (B_t A_t - B_0 A_0) x``.  Both terms receive the
    same dropout realization, so initialization is a numerical no-op.  This is
    equivalent to the paper's residual-base construction without mutating the
    base checkpoint.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float,
        null_basis: torch.Tensor,
        *,
        init_scale: float = 1.0,
        freeze_a: bool = False,
        routing: str = "none",
    ) -> None:
        super().__init__(base_layer, rank, alpha, dropout, routing=routing)
        b_weight, a_weight = factor_projected_weight(
            base_layer.weight,
            null_basis,
            scaling=self.scaling,
            init_scale=init_scale,
        )
        factory_dtype = base_layer.weight.dtype
        factory_device = base_layer.weight.device
        with torch.no_grad():
            self.lora_A.weight.copy_(a_weight.to(device=factory_device, dtype=factory_dtype))
            self.lora_B.weight.copy_(b_weight.to(device=factory_device, dtype=factory_dtype))
        self.register_buffer("anchor_A", self.lora_A.weight.detach().clone(), persistent=True)
        self.register_buffer("anchor_B", self.lora_B.weight.detach().clone(), persistent=True)
        self.register_buffer(
            "projection_basis",
            null_basis.detach().to(device=factory_device, dtype=torch.float32).clone(),
            persistent=False,
        )
        self.init_scale = float(init_scale)
        self.freeze_a = bool(freeze_a)
        self.lora_A.weight.requires_grad = not self.freeze_a
        self.initialization_relative_error = projected_factor_relative_error(
            base_layer.weight,
            null_basis,
            self.anchor_B,
            self.anchor_A,
            scaling=self.scaling,
            init_scale=self.init_scale,
        )
        if self.initialization_relative_error > 5e-4:
            raise RuntimeError(
                "LoRA-Null projected-weight factorization failed: "
                f"relative_error={self.initialization_relative_error:.3e}"
            )

    @torch.no_grad()
    def project_lora_a_(self) -> dict[str, float]:
        """Project trainable A rows back into the calibrated input subspace."""
        current = self.lora_A.weight.detach().float()
        basis = self.projection_basis.to(device=current.device, dtype=torch.float32)
        projected = (current @ basis) @ basis.mT
        total = float(current.square().sum().item())
        outside = float((current - projected).square().sum().item())
        self.lora_A.weight.copy_(projected.to(dtype=self.lora_A.weight.dtype))
        after = self.lora_A.weight.detach().float()
        after_projected = (after @ basis) @ basis.mT
        outside_after = float((after - after_projected).square().sum().item())
        return {
            "total_energy_before": total,
            "outside_energy_before": outside,
            "outside_energy_after": outside_after,
        }

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        if self.scaling == 0.0:
            return base
        adapted_inputs = self.dropout(inputs)
        current = self.lora_B(self.lora_A(adapted_inputs))
        anchor = F.linear(F.linear(adapted_inputs, self.anchor_A), self.anchor_B)
        delta = (current - anchor) * self.scaling
        if self.routing != "none":
            if self.route_mask is None:
                raise RuntimeError(f"LoRA route mask is unset for routing={self.routing}")
            expected = inputs.shape[:-1]
            if tuple(self.route_mask.shape) != tuple(expected):
                raise ValueError(
                    f"LoRA route mask shape {tuple(self.route_mask.shape)} != {tuple(expected)}"
                )
            delta = delta * self.route_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return base + delta

    def merged_weight(self) -> torch.Tensor:
        """Return ``W0 + s * (BtAt - B0A0)`` without changing the module."""

        current = self.lora_B.weight @ self.lora_A.weight
        anchor = self.anchor_B @ self.anchor_A
        return self.base_layer.weight + self.scaling * (current - anchor)


class LoRAEmbedding(nn.Module):
    """Frozen token embedding plus a routed low-rank delta."""

    def __init__(
        self,
        base_layer: nn.Embedding,
        rank: int,
        alpha: float,
        dropout: float,
        routing: str,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.routing = routing
        self.route_mask: torch.Tensor | None = None
        factory = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.lora_A = nn.Embedding(base_layer.num_embeddings, rank, **factory)
        self.lora_B = nn.Linear(rank, base_layer.embedding_dim, bias=False, **factory)
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lora_B.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    @property
    def num_embeddings(self) -> int:
        return self.base_layer.num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self.base_layer.embedding_dim

    @property
    def weight(self) -> torch.Tensor:
        return self.base_layer.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(input_ids)
        delta = self.lora_B(self.dropout(self.lora_A(input_ids))) * self.scaling
        if self.route_mask is None:
            raise RuntimeError(f"embedding route mask is unset for routing={self.routing}")
        if tuple(self.route_mask.shape) != tuple(input_ids.shape):
            raise ValueError("text embedding route mask does not match input tokens")
        route = self.route_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return base + delta * route


class RoutedEmbeddingDelta(nn.Module):
    """Frozen audio embedding plus a zero-initialized target-region delta."""

    def __init__(self, base_layer: nn.Embedding, routing: str):
        super().__init__()
        self.base_layer = base_layer
        self.routing = routing
        self.route_mask: torch.Tensor | None = None
        self.scaling = 1.0
        factory = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.delta = nn.Embedding(
            base_layer.num_embeddings,
            base_layer.embedding_dim,
            **factory,
        )
        nn.init.zeros_(self.delta.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(input_ids)
        delta = self.delta(input_ids) * self.scaling
        if self.route_mask is None:
            raise RuntimeError(f"embedding delta route mask is unset for routing={self.routing}")
        if input_ids.ndim != 3 or tuple(self.route_mask.shape) != (
            input_ids.shape[0], input_ids.shape[2]
        ):
            raise ValueError("embedding delta expects ids [B, C, S] and route [B, S]")
        route = self.route_mask.to(device=delta.device, dtype=delta.dtype)[:, None, :, None]
        return base + delta * route


class RoutedLinearDelta(nn.Module):
    """Frozen output projection plus a zero-initialized full-rank delta."""

    def __init__(self, base_layer: nn.Linear, routing: str):
        super().__init__()
        self.base_layer = base_layer
        self.routing = routing
        self.route_mask: torch.Tensor | None = None
        self.scaling = 1.0
        factory = {"device": base_layer.weight.device, "dtype": base_layer.weight.dtype}
        self.delta = nn.Linear(
            base_layer.in_features,
            base_layer.out_features,
            bias=False,
            **factory,
        )
        nn.init.zeros_(self.delta.weight)
        for parameter in self.base_layer.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(inputs)
        delta = self.delta(inputs) * self.scaling
        if self.route_mask is None:
            raise RuntimeError(f"output delta route mask is unset for routing={self.routing}")
        if tuple(self.route_mask.shape) != tuple(inputs.shape[:-1]):
            raise ValueError("output delta route mask does not match input tokens")
        route = self.route_mask.to(device=delta.device, dtype=delta.dtype).unsqueeze(-1)
        return base + delta * route


class ResidualAdapter(nn.Module):
    """Bottleneck residual branch, initialized as an exact no-op."""

    def __init__(self, hidden_size: int, bottleneck: int, reference: torch.Tensor):
        super().__init__()
        factory = {"device": reference.device, "dtype": reference.dtype}
        self.down = nn.Linear(hidden_size, bottleneck, bias=True, **factory)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, hidden_size, bias=True, **factory)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(hidden_states)))


def _adapter_output_hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> Any:
    adapter = module.peft_adapter
    if isinstance(output, torch.Tensor):
        return output + adapter(output)
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return (output[0] + adapter(output[0]), *output[1:])
    raise TypeError(f"Unsupported Qwen block output type: {type(output)!r}")


def _qwen_layers(model: nn.Module) -> nn.ModuleList:
    llm = getattr(model, "llm", None)
    layers = getattr(llm, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Expected OmniVoice.llm.layers to be torch.nn.ModuleList")
    if len(layers) != 28:
        raise ValueError(f"Expected 28 Qwen3 blocks, found {len(layers)}")
    return layers


def _freeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _wrap_linear(
    projection: nn.Linear,
    module_name: str,
    spec: PeftSpec,
    bases: dict[str, torch.Tensor] | None,
) -> LoRALinear:
    if spec.initialization == "vanilla":
        return LoRALinear(
            projection,
            spec.rank,
            spec.alpha,
            spec.dropout,
            routing=spec.routing,
        )
    if bases is None or module_name not in bases:
        raise KeyError(f"null-basis cache lacks target module {module_name!r}")
    basis = bases[module_name]
    if tuple(basis.shape) != (projection.in_features, spec.rank):
        raise ValueError(
            f"basis {module_name} has shape {tuple(basis.shape)}, expected "
            f"({projection.in_features}, {spec.rank})"
        )
    return LoRANullLinear(
        projection,
        spec.rank,
        spec.alpha,
        spec.dropout,
        basis,
        init_scale=spec.null_init_scale,
        freeze_a=spec.freeze_a,
        routing=spec.routing,
    )


def _inject_lora(
    model: nn.Module,
    spec: PeftSpec,
    bases: dict[str, torch.Tensor] | None = None,
) -> list[str]:
    replaced: list[str] = []
    layers = _qwen_layers(model)
    block_targets = tuple(
        name for name in spec.target_modules if name not in ROOT_PROJECTIONS
    )
    for layer_index in spec.layer_indices:
        block = layers[layer_index]
        attention = getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None)
        if attention is None or mlp is None:
            raise TypeError(f"Qwen block {layer_index} lacks self_attn or mlp")
        for projection_name in block_targets:
            owner = attention if projection_name in TARGET_PROJECTIONS else mlp
            owner_name = "self_attn" if projection_name in TARGET_PROJECTIONS else "mlp"
            projection = getattr(owner, projection_name, None)
            if not isinstance(projection, nn.Linear):
                raise TypeError(
                    f"llm.layers.{layer_index}.{owner_name}.{projection_name} "
                    f"must be nn.Linear, found {type(projection)!r}"
                )
            module_name = f"llm.layers.{layer_index}.{owner_name}.{projection_name}"
            wrapped = _wrap_linear(projection, module_name, spec, bases)
            setattr(owner, projection_name, wrapped)
            replaced.append(module_name)
    if "audio_heads" in spec.target_modules:
        projection = getattr(model, "audio_heads", None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(
                f"audio_heads must be nn.Linear, found {type(projection)!r}"
            )
        model.audio_heads = _wrap_linear(projection, "audio_heads", spec, bases)
        replaced.append("audio_heads")
    if "text_embeddings" in spec.target_modules:
        embedding = model.get_input_embeddings()
        if not isinstance(embedding, nn.Embedding):
            raise TypeError(
                f"text input embeddings must be nn.Embedding, found {type(embedding)!r}"
            )
        model.set_input_embeddings(
            LoRAEmbedding(
                embedding,
                spec.rank,
                spec.alpha,
                spec.dropout,
                routing=spec.routing,
            )
        )
        replaced.append("text_embeddings")
    if "audio_embeddings_delta" in spec.target_modules:
        embedding = getattr(model, "audio_embeddings", None)
        if not isinstance(embedding, nn.Embedding):
            raise TypeError(f"audio_embeddings must be nn.Embedding, found {type(embedding)!r}")
        model.audio_embeddings = RoutedEmbeddingDelta(embedding, routing=spec.routing)
        replaced.append("audio_embeddings_delta")
    if "audio_heads_delta" in spec.target_modules:
        if "audio_heads" in spec.target_modules:
            raise ValueError("audio_heads and audio_heads_delta are mutually exclusive")
        projection = getattr(model, "audio_heads", None)
        if not isinstance(projection, nn.Linear):
            raise TypeError(f"audio_heads must be nn.Linear, found {type(projection)!r}")
        model.audio_heads = RoutedLinearDelta(projection, routing=spec.routing)
        replaced.append("audio_heads_delta")
    return replaced


def set_routing_mask(model: nn.Module, route_mask: torch.Tensor | None) -> int:
    """Set one [batch, sequence] token mask on every routed LoRA projection."""
    routed = 0
    routed_types = (LoRALinear, LoRAEmbedding, RoutedEmbeddingDelta, RoutedLinearDelta)
    for module in model.modules():
        if isinstance(module, routed_types) and module.routing != "none":
            module.route_mask = route_mask
            routed += 1
    if routed and route_mask is None:
        raise ValueError("routed LoRA requires a non-null route mask")
    return routed


def inference_routing_mask(
    *,
    routing: str,
    input_ids: torch.Tensor,
    audio_mask: torch.Tensor,
    audio_mask_id: int,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Build the exact generation-time route for conditional/unconditional rows."""
    if routing == "none":
        return None
    if routing == "prediction":
        return (input_ids == audio_mask_id).any(dim=1) & audio_mask.bool()
    if routing == "text_only":
        return ~audio_mask.bool()
    if routing == "text_prediction":
        prediction = (input_ids == audio_mask_id).any(dim=1) & audio_mask.bool()
        return (~audio_mask.bool()) | prediction
    if routing != "target_region":
        raise ValueError(f"unsupported LoRA routing mode: {routing}")

    batch_size, sequence_length = audio_mask.shape
    if batch_size % 2:
        raise ValueError("target-region inference expects paired conditional/unconditional rows")
    half = batch_size // 2
    target_lengths = audio_mask[half:].sum(dim=1).to(dtype=torch.long)
    if attention_mask is None:
        conditional_lengths = torch.full_like(target_lengths, sequence_length)
    elif attention_mask.ndim == 4:
        conditional_lengths = attention_mask[:half, 0, 0].sum(dim=-1).to(dtype=torch.long)
    elif attention_mask.ndim == 2:
        conditional_lengths = attention_mask[:half].sum(dim=-1).to(dtype=torch.long)
    else:
        raise ValueError(f"unsupported attention mask rank: {attention_mask.ndim}")

    route = torch.zeros_like(audio_mask, dtype=torch.bool)
    for index, (conditional_length, target_length) in enumerate(
        zip(conditional_lengths.tolist(), target_lengths.tolist())
    ):
        if target_length <= 0 or target_length > conditional_length:
            raise ValueError(
                f"invalid target length {target_length} for conditional length {conditional_length}"
            )
        route[index, conditional_length - target_length : conditional_length] = True
        route[half + index, :target_length] = True
    return route


def _inject_residual_adapters(model: nn.Module, spec: PeftSpec) -> list[str]:
    layers = _qwen_layers(model)
    first_layer = len(layers) - spec.adapter_layers
    injected: list[str] = []
    for layer_index in range(first_layer, len(layers)):
        block = layers[layer_index]
        if hasattr(block, "peft_adapter"):
            raise RuntimeError(f"Qwen block {layer_index} already has peft_adapter")
        reference = next(block.parameters())
        hidden_size = model.config.llm_config.hidden_size
        block.add_module(
            "peft_adapter",
            ResidualAdapter(hidden_size, spec.bottleneck, reference),
        )
        block._peft_adapter_hook_handle = block.register_forward_hook(
            _adapter_output_hook
        )
        injected.append(f"llm.layers.{layer_index}.peft_adapter")
    return injected


def inject_peft(model: nn.Module, spec: PeftSpec) -> dict[str, Any]:
    """Freeze the complete base model and inject exactly one E4/E5 strategy."""
    spec.validate()
    if hasattr(model, "_local_peft_spec"):
        raise RuntimeError("PEFT has already been injected into this model")
    _freeze_all(model)
    if spec.strategy == "lora":
        bases = None
        basis_metadata = None
        if spec.initialization == "lora_null":
            bases, basis_metadata = load_basis_cache(
                spec.null_basis_path,
                expected_sha256=spec.null_basis_sha256,
            )
            observed_base = basis_metadata.get("base_model_sha256")
            if observed_base != spec.base_model_sha256:
                raise RuntimeError(
                    "null-basis base model mismatch: "
                    f"cache={observed_base!r}, adapter={spec.base_model_sha256!r}"
                )
            observed_rank = basis_metadata.get("rank")
            if observed_rank is not None and int(observed_rank) != spec.rank:
                raise RuntimeError(
                    f"null-basis rank {observed_rank} != adapter rank {spec.rank}"
                )
        modules = _inject_lora(model, spec, bases=bases)
        if basis_metadata is not None:
            model._null_basis_metadata = basis_metadata
    else:
        modules = _inject_residual_adapters(model, spec)
    model._local_peft_spec = spec
    report = audit_trainable_parameters(model, spec)
    report["injected_modules"] = modules
    return report


@torch.no_grad()
def project_lora_null_a(model: nn.Module) -> dict[str, float | int]:
    """Apply the exact post-step A projection to every LoRA-Null module."""
    modules = [module for module in model.modules() if isinstance(module, LoRANullLinear)]
    if not modules:
        raise RuntimeError("hard_project_a requested but no LoRA-Null modules were found")
    total = 0.0
    outside_before = 0.0
    outside_after = 0.0
    for module in modules:
        result = module.project_lora_a_()
        total += result["total_energy_before"]
        outside_before += result["outside_energy_before"]
        outside_after += result["outside_energy_after"]
    return {
        "modules": len(modules),
        "outside_energy_fraction_before": outside_before / max(total, 1e-30),
        "outside_norm_fraction_before": math.sqrt(outside_before / max(total, 1e-30)),
        "outside_energy_fraction_after": outside_after / max(total, 1e-30),
    }


def _is_adapter_parameter(name: str, strategy: str) -> bool:
    if strategy == "lora":
        return (
            ".lora_A.weight" in name
            or ".lora_B.weight" in name
            or name.endswith(".delta.weight")
        )
    return ".peft_adapter.down." in name or ".peft_adapter.up." in name


def _is_adapter_state(name: str, strategy: str) -> bool:
    return _is_adapter_parameter(name, strategy) or (
        strategy == "lora" and (name.endswith(".anchor_A") or name.endswith(".anchor_B"))
    )


def adapter_state_dict(model: nn.Module, spec: PeftSpec) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous()
        for name, value in model.state_dict().items()
        if _is_adapter_state(name, spec.strategy)
    }


def audit_trainable_parameters(model: nn.Module, spec: PeftSpec) -> dict[str, Any]:
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    unexpected = [name for name, _ in trainable if not _is_adapter_parameter(name, spec.strategy)]
    if unexpected:
        raise AssertionError(f"Unexpected trainable base parameters: {unexpected}")
    frozen_guards = {}
    embeddings = getattr(model, "audio_embeddings", None)
    if embeddings is None:
        raise AssertionError("OmniVoice is missing audio_embeddings")
    embedding_base = (
        embeddings.base_layer if isinstance(embeddings, RoutedEmbeddingDelta) else embeddings
    )
    frozen_guards["audio_embeddings"] = all(
        not parameter.requires_grad for parameter in embedding_base.parameters()
    )
    if not frozen_guards["audio_embeddings"]:
        raise AssertionError("audio_embeddings must remain frozen")
    audio_heads = getattr(model, "audio_heads", None)
    if audio_heads is None:
        raise AssertionError("OmniVoice is missing audio_heads")
    head_base = (
        audio_heads.base_layer
        if isinstance(audio_heads, (LoRALinear, RoutedLinearDelta))
        else audio_heads
    )
    frozen_guards["audio_heads_base"] = all(
        not parameter.requires_grad for parameter in head_base.parameters()
    )
    if not frozen_guards["audio_heads_base"]:
        raise AssertionError("the base audio_heads parameters must remain frozen")

    text_embeddings = model.get_input_embeddings()
    text_embedding_base = (
        text_embeddings.base_layer
        if isinstance(text_embeddings, LoRAEmbedding)
        else text_embeddings
    )
    frozen_guards["text_embeddings_base"] = all(
        not parameter.requires_grad for parameter in text_embedding_base.parameters()
    )
    if not frozen_guards["text_embeddings_base"]:
        raise AssertionError("the base text embeddings must remain frozen")

    if spec.strategy == "lora":
        lora_modules = [
            module for module in model.modules()
            if isinstance(module, (LoRALinear, LoRAEmbedding))
        ]
        delta_modules = [
            module
            for module in model.modules()
            if isinstance(module, (RoutedEmbeddingDelta, RoutedLinearDelta))
        ]
        block_target_count = sum(
            name not in ROOT_PROJECTIONS for name in spec.target_modules
        )
        root_target_count = sum(
            name in ROOT_PROJECTIONS for name in spec.target_modules
        )
        expected_modules = len(spec.layer_indices) * block_target_count + root_target_count
        if len(lora_modules) + len(delta_modules) != expected_modules:
            raise AssertionError(
                f"Expected {expected_modules} adapter modules, "
                f"found {len(lora_modules) + len(delta_modules)}"
            )
        expected_tensors = sum(
            int(module.lora_A.weight.requires_grad) + int(module.lora_B.weight.requires_grad)
            for module in lora_modules
        ) + len(delta_modules)
        expected_parameters = sum(
            (module.lora_A.weight.numel() if module.lora_A.weight.requires_grad else 0)
            + (module.lora_B.weight.numel() if module.lora_B.weight.requires_grad else 0)
            for module in lora_modules
        ) + sum(module.delta.weight.numel() for module in delta_modules)
    else:
        injected_modules = [module for module in model.modules() if isinstance(module, ResidualAdapter)]
        if len(injected_modules) != 4:
            raise AssertionError(f"Expected 4 residual adapters, found {len(injected_modules)}")
        expected_tensors = len(injected_modules) * 4
        expected_parameters = sum(
            sum(parameter.numel() for parameter in module.parameters())
            for module in injected_modules
        )
    count = sum(parameter.numel() for _, parameter in trainable)
    if len(trainable) != expected_tensors or count != expected_parameters:
        raise AssertionError(
            f"Trainable contract mismatch: tensors={len(trainable)}/{expected_tensors}, "
            f"parameters={count}/{expected_parameters}"
        )
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "experiment_id": spec.experiment_id,
        "strategy": spec.strategy,
        "trainable_tensors": len(trainable),
        "trainable_parameters": count,
        "total_parameters_after_injection": total,
        "trainable_percent": 100.0 * count / total,
        "frozen_guards": frozen_guards,
        "trainable_names": [name for name, _ in trainable],
    }


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("No trainable PEFT parameters found")
    return parameters


def save_adapter_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    spec: PeftSpec,
    audit: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state = adapter_state_dict(model, spec)
    expected = audit_trainable_parameters(model, spec)
    if not set(expected["trainable_names"]).issubset(state):
        raise AssertionError("Adapter state is missing trainable parameters")
    save_file(state, checkpoint_dir / ADAPTER_FILE)
    (checkpoint_dir / PEFT_CONFIG_FILE).write_text(
        json.dumps(spec.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (checkpoint_dir / AUDIT_FILE).write_text(
        json.dumps(audit or expected, indent=2) + "\n", encoding="utf-8"
    )
    if metadata is not None:
        (checkpoint_dir / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )


def read_peft_spec(checkpoint_dir: str | Path) -> PeftSpec:
    path = Path(checkpoint_dir) / PEFT_CONFIG_FILE
    return PeftSpec.from_dict(json.loads(path.read_text(encoding="utf-8")))


def load_adapter_checkpoint(
    model: nn.Module,
    checkpoint_dir: str | Path,
    *,
    inject_if_needed: bool = True,
    trainable: bool = False,
) -> dict[str, Any]:
    """Restore adapter weights into a clean or already-injected base model."""
    checkpoint_dir = Path(checkpoint_dir)
    spec = read_peft_spec(checkpoint_dir)
    current_spec = getattr(model, "_local_peft_spec", None)
    if current_spec is None:
        if not inject_if_needed:
            raise RuntimeError("Model has no injected adapter")
        inject_peft(model, spec)
    elif current_spec != spec:
        raise ValueError(f"Injected spec {current_spec} differs from checkpoint {spec}")
    state = load_file(checkpoint_dir / ADAPTER_FILE, device="cpu")
    expected_names = set(adapter_state_dict(model, spec))
    if set(state) != expected_names:
        missing = sorted(expected_names - set(state))
        extra = sorted(set(state) - expected_names)
        raise RuntimeError(f"Adapter key mismatch; missing={missing}, extra={extra}")
    state_targets = model.state_dict(keep_vars=True)
    originally_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    with torch.no_grad():
        for name, value in state.items():
            target = state_targets[name]
            target.copy_(value.to(device=target.device, dtype=target.dtype))
    for name, parameter in model.named_parameters():
        parameter.requires_grad = trainable and name in originally_trainable
    audit = audit_trainable_parameters(model, spec) if trainable else {
        "experiment_id": spec.experiment_id,
        "strategy": spec.strategy,
        "loaded_tensors": len(state),
        "loaded_parameters": sum(value.numel() for value in state.values()),
        "all_parameters_frozen": all(not p.requires_grad for p in model.parameters()),
    }
    return audit
