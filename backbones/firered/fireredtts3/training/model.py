"""Teacher-forced FireRedTTS3 training objective.

The released model predicts one RedAE patch at a time.  At generation step
``j`` the flow head receives clean history latents, a noised current patch,
speaker conditioning, and the last ``history_patches + 1`` causal backbone
states.  This module reproduces that alignment without exposing a future target
patch to its own flow prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .lora import (
    ROUTING_GLOBAL,
    build_backbone_route_mask,
    set_lora_route_mask,
    uses_routed_lora,
)


@dataclass
class FireRedTTS3TrainingOutput:
    loss: torch.Tensor
    flow_loss: torch.Tensor
    stop_loss: torch.Tensor
    stop_positive_probability: torch.Tensor
    stop_max_negative_probability: torch.Tensor
    stop_logit_margin: torch.Tensor
    effective_stop_positive_weight: float
    sampled_flow_patches: int
    target_patches: int


def resolve_stop_positive_weight(
    setting: float | str,
    *,
    negative_targets: int,
    maximum: float,
) -> float:
    """Resolve a fixed or per-utterance class-balance weight.

    Each target contains exactly one positive end decision and
    ``negative_targets`` continuation decisions.  ``auto`` therefore follows
    the standard negative/positive ratio, with a configurable safety cap.
    """

    if negative_targets <= 0:
        raise ValueError("stop loss requires at least one negative target")
    if maximum <= 0:
        raise ValueError("stop_positive_weight_max must be positive")
    if isinstance(setting, str):
        if setting != "auto":
            raise ValueError("stop_positive_weight must be a positive number or 'auto'")
        return min(float(negative_targets), float(maximum))
    value = float(setting)
    if not torch.isfinite(torch.tensor(value)) or value <= 0:
        raise ValueError("stop_positive_weight must be finite and positive")
    return min(value, float(maximum))


def balanced_stop_loss(
    stop_logits: torch.Tensor,
    *,
    positive_weight: float | str,
    positive_weight_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Compute length-balanced stop BCE plus observable decision statistics."""

    if stop_logits.ndim != 1 or stop_logits.numel() < 2:
        raise ValueError("stop logits must contain continuation decisions and one end decision")
    stop_targets = torch.zeros_like(stop_logits)
    stop_targets[-1] = 1.0
    effective_weight = resolve_stop_positive_weight(
        positive_weight,
        negative_targets=stop_logits.numel() - 1,
        maximum=positive_weight_max,
    )
    stop_loss = F.binary_cross_entropy_with_logits(
        stop_logits,
        stop_targets,
        pos_weight=stop_logits.new_tensor(effective_weight),
    )
    probabilities = stop_logits.sigmoid()
    positive_probability = probabilities[-1]
    max_negative_probability = probabilities[:-1].max()
    logit_margin = stop_logits[-1] - stop_logits[:-1].max()
    return (
        stop_loss,
        positive_probability,
        max_negative_probability,
        logit_margin,
        effective_weight,
    )


def sample_patch_indices(
    target_patches: int,
    max_flow_patches: int | None,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if target_patches <= 0:
        raise ValueError("target audio must contain at least one patch")
    if max_flow_patches is None or max_flow_patches <= 0 or target_patches <= max_flow_patches:
        return torch.arange(target_patches, device=device)
    selected = torch.randperm(target_patches, device=device, generator=generator)[:max_flow_patches]
    return selected.sort().values


def build_condition_windows(
    audio_hidden: torch.Tensor,
    *,
    prompt_patches: int,
    target_patch_indices: torch.Tensor,
    history_patches: int,
) -> torch.Tensor:
    """Return causal patch-state windows ending immediately before each target.

    ``audio_hidden`` contains prompt patch states followed by teacher-forced
    target patch states.  For target patch 0 the final state in its condition is
    the last prompt state; for target patch j it is target state j-1.
    """

    if audio_hidden.ndim != 3 or audio_hidden.shape[0] != 1:
        raise ValueError(f"expected audio_hidden shape (1, patches, dim), got {audio_hidden.shape}")
    if prompt_patches <= 0:
        raise ValueError("at least one prompt patch is required")
    left = audio_hidden.new_zeros(1, history_patches, audio_hidden.shape[-1])
    padded = torch.cat([left, audio_hidden], dim=1)[0]
    ends = prompt_patches - 1 + target_patch_indices
    offsets = torch.arange(history_patches + 1, device=audio_hidden.device)
    # Left padding makes padded index ``end`` the beginning of an H+1 window.
    gather = ends.unsqueeze(1) + offsets.unsqueeze(0)
    return padded[gather]


def build_latent_history_windows(
    prompt_latents: torch.Tensor,
    target_latents: torch.Tensor,
    *,
    target_patch_indices: torch.Tensor,
    patch_size: int,
    history_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build clean histories and current clean patches for selected targets."""

    all_latents = torch.cat([prompt_latents, target_latents], dim=1)
    padded = torch.cat(
        [all_latents.new_zeros(1, history_length, all_latents.shape[-1]), all_latents],
        dim=1,
    )[0]
    starts = prompt_latents.shape[1] + target_patch_indices * patch_size
    history_offsets = torch.arange(history_length, device=all_latents.device)
    histories = padded[starts.unsqueeze(1) + history_offsets.unsqueeze(0)]

    target = target_latents[0].reshape(-1, patch_size, target_latents.shape[-1])
    current = target[target_patch_indices]
    return histories, current


class FireRedTTS3ForTraining(nn.Module):
    """Wrap a :class:`FireRedTTS3BaseCore` with flow and stop losses."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        *,
        spk_emb: torch.Tensor,
        text_tokens: torch.Tensor,
        prompt_latents: torch.Tensor,
        target_latents: torch.Tensor,
        sample_weight: float | torch.Tensor = 1.0,
        max_flow_patches: int | None = 24,
        cfg_dropout_prob: float = 0.1,
        stop_loss_weight: float = 0.1,
        stop_positive_weight: float | str = 1.0,
        stop_positive_weight_max: float = 64.0,
        generator: torch.Generator | None = None,
    ) -> FireRedTTS3TrainingOutput:
        core = self.core
        patch_size = int(core.patch_size)
        history_patches = int(core.history_patches)
        history_length = int(core.history_length)
        if prompt_latents.shape[0] != 1 or target_latents.shape[0] != 1:
            raise ValueError("the released PatchEncoder currently requires per-device batch size 1")
        if prompt_latents.shape[1] % patch_size or target_latents.shape[1] % patch_size:
            raise ValueError(
                f"latent lengths must be divisible by patch_size={patch_size}: "
                f"prompt={prompt_latents.shape[1]}, target={target_latents.shape[1]}"
            )

        prompt_patches = prompt_latents.shape[1] // patch_size
        target_patches = target_latents.shape[1] // patch_size
        selected = sample_patch_indices(
            target_patches,
            max_flow_patches,
            device=target_latents.device,
            generator=generator,
        )

        # Full causal teacher forcing.  Target patch j is present in the input,
        # but build_condition_windows ends at j-1, so its flow target cannot leak.
        routing = getattr(core, "_lora_routing", ROUTING_GLOBAL)
        routed = uses_routed_lora(routing)
        if routed:
            patch_route = torch.cat(
                [
                    target_latents.new_zeros(prompt_patches),
                    target_latents.new_ones(target_patches),
                ]
            ).reshape(prompt_patches + target_patches, 1, 1)
            set_lora_route_mask(core.patch_encoder, patch_route)
        else:
            set_lora_route_mask(core.patch_encoder, None)

        text_embeds = core.backbone_llm.embed_tokens(text_tokens)
        audio_embeds = core.patch_encoder(torch.cat([prompt_latents, target_latents], dim=1))
        speaker_token = core.spk_proj_llm(spk_emb).unsqueeze(1)
        input_embeds = torch.cat([speaker_token, text_embeds, audio_embeds], dim=1)
        backbone_route = build_backbone_route_mask(
            routing,
            text_tokens=text_tokens.shape[1],
            prompt_patches=prompt_patches,
            target_patches=target_patches,
            reference=input_embeds,
        )
        set_lora_route_mask(core.backbone_llm, backbone_route)
        # DiT is invoked only for selected target patches, so its adapter is
        # always active even when the causal backbone and PatchEncoder route.
        set_lora_route_mask(core.dit, None)
        backbone_output = core.backbone_llm(
            inputs_embeds=input_embeds,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        audio_start = 1 + text_tokens.shape[1]
        audio_hidden = backbone_output[:, audio_start : audio_start + prompt_patches + target_patches]

        condition_windows = build_condition_windows(
            audio_hidden,
            prompt_patches=prompt_patches,
            target_patch_indices=selected,
            history_patches=history_patches,
        )
        histories, clean_current = build_latent_history_windows(
            prompt_latents,
            target_latents,
            target_patch_indices=selected,
            patch_size=patch_size,
            history_length=history_length,
        )

        noise = torch.randn(
            clean_current.shape,
            device=clean_current.device,
            dtype=clean_current.dtype,
            generator=generator,
        )
        times = torch.rand(
            clean_current.shape[0],
            device=clean_current.device,
            dtype=torch.float32,
            generator=generator,
        )
        time_view = times.to(clean_current.dtype).view(-1, 1, 1)
        noised_current = (1.0 - time_view) * noise + time_view * clean_current
        latent_input = torch.cat([histories, noised_current], dim=1)

        backbone_condition = core.dit_head(condition_windows).repeat_interleave(patch_size, dim=1)
        speaker_condition = core.spk_proj_dit(spk_emb).expand(selected.numel(), -1)
        speaker_condition = speaker_condition.unsqueeze(1).expand(-1, latent_input.shape[1], -1)
        condition = torch.cat([backbone_condition, speaker_condition], dim=-1)
        if cfg_dropout_prob > 0:
            dropped = torch.rand(
                selected.numel(), 1, 1,
                device=condition.device,
                generator=generator,
            ) < cfg_dropout_prob
            condition = condition.masked_fill(dropped, 0.0)

        velocity_prediction = core.dit(torch.cat([latent_input, condition], dim=-1), times)
        velocity_target = clean_current - noise
        flow_loss = F.mse_loss(
            velocity_prediction[:, -patch_size:].float(),
            velocity_target.float(),
            reduction="none",
        ).mean(dim=(1, 2)).mean()

        # Generation decisions are made at last-prompt, target-0, ..., target-last.
        stop_hidden = audio_hidden[:, prompt_patches - 1 : prompt_patches + target_patches]
        stop_logits = core.stop_head(stop_hidden).squeeze(0).squeeze(-1).float()
        (
            stop_loss,
            stop_positive_probability,
            stop_max_negative_probability,
            stop_logit_margin,
            effective_stop_positive_weight,
        ) = balanced_stop_loss(
            stop_logits,
            positive_weight=stop_positive_weight,
            positive_weight_max=stop_positive_weight_max,
        )

        weight = torch.as_tensor(sample_weight, device=flow_loss.device, dtype=flow_loss.dtype)
        loss = weight * (flow_loss + float(stop_loss_weight) * stop_loss)
        return FireRedTTS3TrainingOutput(
            loss=loss,
            flow_loss=flow_loss,
            stop_loss=stop_loss,
            stop_positive_probability=stop_positive_probability,
            stop_max_negative_probability=stop_max_negative_probability,
            stop_logit_margin=stop_logit_margin,
            effective_stop_positive_weight=effective_stop_positive_weight,
            sampled_flow_patches=int(selected.numel()),
            target_patches=int(target_patches),
        )


def output_as_dict(output: FireRedTTS3TrainingOutput) -> dict[str, Any]:
    return {
        "loss": float(output.loss.detach()),
        "flow_loss": float(output.flow_loss.detach()),
        "stop_loss": float(output.stop_loss.detach()),
        "stop_positive_probability": float(output.stop_positive_probability.detach()),
        "stop_max_negative_probability": float(output.stop_max_negative_probability.detach()),
        "stop_logit_margin": float(output.stop_logit_margin.detach()),
        "effective_stop_positive_weight": output.effective_stop_positive_weight,
        "sampled_flow_patches": output.sampled_flow_patches,
        "target_patches": output.target_patches,
    }
