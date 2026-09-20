"""Equation (3), retaining each backbone's within-example objective."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def weighted_example_mean(sample_losses, sample_weights):
    if sample_losses.ndim != 1 or sample_losses.numel() == 0:
        raise ValueError("sample_losses must be a nonempty [B] vector")
    weights = torch.as_tensor(sample_weights, device=sample_losses.device, dtype=sample_losses.dtype)
    if weights.shape != sample_losses.shape:
        raise ValueError("one weight is required per example; broadcasting is forbidden")
    if not torch.isfinite(weights).all() or torch.any((weights < 0) | (weights > 1)):
        raise ValueError("sample weights must be finite and in [0, 1]")
    if not torch.isfinite(sample_losses).all():
        raise ValueError("sample losses must be finite")
    return (sample_losses * weights).mean()


def firered_loss(flow_per_example, stop_per_example, sample_weights, stop_weight=0.1):
    if flow_per_example.shape != stop_per_example.shape:
        raise ValueError("flow and stop losses must have matching [B] shapes")
    return weighted_example_mean(flow_per_example + stop_weight * stop_per_example, sample_weights)


def omnivoice_loss(logits, labels, sample_weights, codebook_weights):
    """[B,C,T,V] logits, [B,C,T] labels; ignore -100, average per example."""
    if logits.ndim != 4 or labels.shape != logits.shape[:-1]:
        raise ValueError("expected logits [B,C,T,V] and labels [B,C,T]")
    codebook = torch.as_tensor(codebook_weights, device=logits.device, dtype=torch.float32)
    if codebook.shape != (logits.shape[1],) or not torch.isfinite(codebook).all() or \
            (codebook < 0).any() or codebook.sum() <= 0:
        raise ValueError("invalid backbone codebook weights")
    valid = labels != -100
    per_token = F.cross_entropy(logits.float().permute(0, 3, 1, 2), labels,
                                reduction="none", ignore_index=-100)
    # Every utterance has equal standing regardless of its duration/token count.
    per_codebook = (per_token * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    per_example = (per_codebook * (codebook / codebook.sum())).sum(-1)
    # A zero-supervision example contributes zero and still occupies a batch slot.
    return weighted_example_mean(per_example, sample_weights)
