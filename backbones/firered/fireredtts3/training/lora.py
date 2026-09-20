"""Small native LoRA implementation used by the FireRed trainer.

Keeping this local avoids a PEFT/Transformers version constraint in the
upstream inference environment.  Only ``torch.nn.Linear`` modules are wrapped.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import nn


ROUTING_GLOBAL = "global"
ROUTING_TARGET_AUDIO = "target_audio"
ROUTING_SEMANTIC_DECODE = "semantic_decode"
SUPPORTED_ROUTING = {
    ROUTING_GLOBAL,
    ROUTING_TARGET_AUDIO,
    ROUTING_SEMANTIC_DECODE,
}


DEFAULT_BACKBONE_PATTERNS = (
    r"^backbone_llm\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$",
)
DEFAULT_DIT_PATTERNS = (
    r"^dit\.blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)$",
    r"^dit\.blocks\.\d+\.mlp\.ff\.(0\.0|2)$",
)
DEFAULT_PATCH_ENCODER_PATTERNS = (
    r"^patch_encoder\.blocks\.\d+\.attn\.(to_q|to_k|to_v|to_out\.0)$",
    r"^patch_encoder\.blocks\.\d+\.mlp\.ff\.(0\.0|2)$",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_patterns: tuple[str, ...] = DEFAULT_BACKBONE_PATTERNS + DEFAULT_DIT_PATTERNS
    train_extra_patterns: tuple[str, ...] = ()
    routing: str = ROUTING_GLOBAL

    def __post_init__(self) -> None:
        if self.routing not in SUPPORTED_ROUTING:
            raise ValueError(
                f"unsupported LoRA routing={self.routing!r}; expected one of {sorted(SUPPORTED_ROUTING)}"
            )

    @classmethod
    def from_json(cls, path: str | Path) -> "LoRAConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            rank=int(value["rank"]),
            alpha=float(value["alpha"]),
            dropout=float(value.get("dropout", 0.0)),
            target_patterns=tuple(value["target_patterns"]),
            train_extra_patterns=tuple(value.get("train_extra_patterns", ())),
            routing=str(value.get("routing", ROUTING_GLOBAL)),
        )

    def save_json(self, path: str | Path) -> None:
        value = asdict(self)
        value["target_patterns"] = list(self.target_patterns)
        value["train_extra_patterns"] = list(self.train_extra_patterns)
        Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class LoRALinear(nn.Module):
    """Frozen linear layer plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        # This is deliberately not a registered buffer: the mask is an
        # ephemeral per-batch routing decision and must never enter adapters.
        self.route_mask: torch.Tensor | None = None
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        update = torch.nn.functional.linear(self.dropout(inputs), self.lora_A)
        update = torch.nn.functional.linear(update, self.lora_B)
        if self.route_mask is not None:
            try:
                update = update * self.route_mask.to(device=update.device, dtype=update.dtype)
            except RuntimeError as error:
                raise RuntimeError(
                    f"LoRA route mask shape {tuple(self.route_mask.shape)} is not broadcastable "
                    f"to update shape {tuple(update.shape)}"
                ) from error
        return base_output + update.to(base_output.dtype) * self.scaling


def set_lora_route_mask(module: nn.Module, mask: torch.Tensor | None) -> int:
    """Apply an ephemeral routing mask to every LoRA wrapper below ``module``.

    The mask multiplies only the low-rank residual.  A zero value therefore
    gives the exact frozen-base output, while a one enables the adapter.  The
    assigned tensor intentionally remains on each wrapper until the next
    forward so non-reentrant gradient-checkpoint recomputation sees the same
    route used by the original forward.
    """

    count = 0
    for child in module.modules():
        if isinstance(child, LoRALinear):
            child.route_mask = mask
            count += 1
    return count


def uses_routed_lora(routing: str) -> bool:
    """Return whether ``routing`` needs explicit per-token masks."""

    if routing not in SUPPORTED_ROUTING:
        raise ValueError(
            f"unsupported LoRA routing={routing!r}; expected one of {sorted(SUPPORTED_ROUTING)}"
        )
    return routing != ROUTING_GLOBAL


def build_backbone_route_mask(
    routing: str,
    *,
    text_tokens: int,
    prompt_patches: int,
    target_patches: int,
    reference: torch.Tensor,
) -> torch.Tensor | None:
    """Build the backbone mask shared by teacher forcing and AR inference.

    Sequence layout is ``speaker, text, prompt audio, target audio``.  The
    legacy ``target_audio`` policy is intentionally preserved for checkpoint
    compatibility.  ``semantic_decode`` keeps prompt-audio K/V projections on
    the frozen base path, while enabling the text positions, the final prompt
    position used as the first decode query, and all generated target tokens.
    This prevents the first AR patch from being conditioned on an entirely
    unadapted Burmese-text representation.
    """

    if min(text_tokens, prompt_patches, target_patches) < 0:
        raise ValueError("route segment lengths must be non-negative")
    if routing == ROUTING_GLOBAL:
        return None
    if routing not in SUPPORTED_ROUTING:
        raise ValueError(
            f"unsupported LoRA routing={routing!r}; expected one of {sorted(SUPPORTED_ROUTING)}"
        )

    total = 1 + text_tokens + prompt_patches + target_patches
    mask = reference.new_zeros((1, total, 1))
    target_start = 1 + text_tokens + prompt_patches
    if target_patches:
        mask[:, target_start:, :] = 1
    if routing == ROUTING_SEMANTIC_DECODE:
        if text_tokens:
            mask[:, 1 : 1 + text_tokens, :] = 1
        if prompt_patches:
            # Generation step zero queries the last prompt state.  Activating
            # only this prompt position gives the adapter a decode anchor while
            # leaving the rest of the acoustic prompt on the base-model path.
            mask[:, target_start - 1 : target_start, :] = 1
    return mask


def _resolve_parent(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    path = name.split(".")
    parent = root
    for component in path[:-1]:
        parent = parent[int(component)] if component.isdigit() else getattr(parent, component)
    return parent, path[-1]


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def freeze_all(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def inject_lora(module: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze ``module`` and replace matching linears with LoRA wrappers."""

    freeze_all(module)
    candidates = list(module.named_modules())
    replaced: list[str] = []
    for name, child in candidates:
        if not name or isinstance(child, LoRALinear):
            continue
        if not isinstance(child, nn.Linear) or not _matches(name, config.target_patterns):
            continue
        parent, attribute = _resolve_parent(module, name)
        wrapped = LoRALinear(
            child,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        if attribute.isdigit():
            parent[int(attribute)] = wrapped
        else:
            setattr(parent, attribute, wrapped)
        replaced.append(name)

    if not replaced:
        raise ValueError(
            "no Linear modules matched LoRA target patterns:\n  "
            + "\n  ".join(config.target_patterns)
        )

    for name, parameter in module.named_parameters():
        if _matches(name, config.train_extra_patterns):
            # Extra trainable base parameters need the same FP32 master-weight
            # guarantee as LoRA A/B; the frozen base itself may be stored BF16.
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
            parameter.requires_grad_(True)
    module._lora_routing = config.routing
    return replaced


def trainable_parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return trainable, total


def trainable_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, value in module.named_parameters() if value.requires_grad}
    return {
        name: value.detach().cpu()
        for name, value in module.state_dict().items()
        if name in trainable_names
    }


def save_lora_adapter(
    core: nn.Module,
    output_dir: str | Path,
    config: LoRAConfig,
    *,
    metadata: dict | None = None,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config.save_json(output / "adapter_config.json")
    state = trainable_state_dict(core)
    temporary = output / "adapter.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(output / "adapter.pt")
    if metadata is not None:
        (output / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def load_lora_adapter(
    core: nn.Module,
    adapter_dir: str | Path,
    *,
    strict: bool = True,
) -> LoRAConfig:
    """Inject and load a saved adapter into an inference or training core."""

    adapter = Path(adapter_dir)
    config = LoRAConfig.from_json(adapter / "adapter_config.json")
    inject_lora(core, config)
    state = torch.load(adapter / "adapter.pt", map_location="cpu", weights_only=True)
    missing, unexpected = core.load_state_dict(state, strict=False)
    adapter_names = set(state)
    missing_trainable = [
        name
        for name, parameter in core.named_parameters()
        if parameter.requires_grad and name not in adapter_names
    ]
    if strict and (unexpected or missing_trainable):
        raise RuntimeError(
            f"adapter load mismatch: unexpected={unexpected}, "
            f"missing_trainable={missing_trainable}; all_missing_count={len(missing)}"
        )
    core._lora_routing = config.routing
    return config


def named_trainable_parameters(module: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            yield name, parameter
