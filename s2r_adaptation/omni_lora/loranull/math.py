"""Numerically stable linear algebra used by the OmniVoice LoRA-Null port.

This is a clean-room implementation of the equations in LoRA-Null.  It avoids
forming the dense projector ``U U.T``: only ``W @ U`` (out_features by rank)
is decomposed.
"""

from __future__ import annotations

import torch


def _check_rank(rank: int, dimension: int) -> None:
    if rank <= 0 or rank > dimension:
        raise ValueError(f"rank must be in [1, {dimension}], got {rank}")


def smallest_eigenbasis(
    covariance: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the ``rank`` lowest-energy orthonormal directions of ``X.T @ X``.

    ``torch.linalg.eigh`` returns eigenvalues in ascending order, so this is the
    covariance equivalent of taking the trailing left singular vectors of the
    paper's activation matrix ``X_pre``.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix")
    _check_rank(rank, covariance.shape[0])
    work = covariance.to(dtype=torch.float64)
    work = (work + work.mT) * 0.5
    eigenvalues, eigenvectors = torch.linalg.eigh(work)
    basis = eigenvectors[:, :rank].to(dtype=torch.float32).contiguous()
    values = eigenvalues[:rank].clamp_min(0).to(dtype=torch.float64).contiguous()
    return basis, values


def factor_projected_weight(
    weight: torch.Tensor,
    null_basis: torch.Tensor,
    *,
    scaling: float,
    init_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor ``(init_scale/scaling) * W U U.T`` into LoRA ``B @ A``.

    The returned shapes are ``B=[out, rank]`` and ``A=[rank, in]``.  With the
    centered forward ``W x + scaling * (B_t A_t - B_0 A_0) x``, initialization
    is an exact no-op while the paper's projected component has magnitude
    ``init_scale`` after LoRA scaling is applied.
    """

    if weight.ndim != 2 or null_basis.ndim != 2:
        raise ValueError("weight and null_basis must both be matrices")
    out_features, in_features = weight.shape
    if null_basis.shape[0] != in_features:
        raise ValueError(
            f"basis input dimension {null_basis.shape[0]} != weight input {in_features}"
        )
    rank = null_basis.shape[1]
    _check_rank(rank, min(in_features, out_features))
    if scaling <= 0 or init_scale < 0:
        raise ValueError("scaling must be positive and init_scale non-negative")

    work_weight = weight.detach().to(dtype=torch.float32)
    basis = null_basis.detach().to(device=work_weight.device, dtype=torch.float32)
    gram = basis.mT @ basis
    identity = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    if not torch.allclose(gram, identity, atol=2e-4, rtol=2e-4):
        raise ValueError("null_basis columns must be orthonormal")

    projected_thin = work_weight @ basis
    left, singular, right_t = torch.linalg.svd(projected_thin, full_matrices=False)
    singular = singular[:rank].clamp_min(0) * (float(init_scale) / float(scaling))
    root = singular.sqrt()
    b_weight = left[:, :rank] * root.unsqueeze(0)
    a_weight = (root.unsqueeze(1) * right_t[:rank]) @ basis.mT
    return b_weight.contiguous(), a_weight.contiguous()


def projected_factor_relative_error(
    weight: torch.Tensor,
    null_basis: torch.Tensor,
    b_weight: torch.Tensor,
    a_weight: torch.Tensor,
    *,
    scaling: float,
    init_scale: float = 1.0,
) -> float:
    """Diagnostic relative error for ``scaling * BA == init_scale * WUU.T``."""

    work = weight.detach().to(dtype=torch.float64)
    basis = null_basis.detach().to(device=work.device, dtype=torch.float64)
    expected = float(init_scale) * (work @ basis) @ basis.mT
    observed = float(scaling) * (
        b_weight.detach().to(device=work.device, dtype=torch.float64)
        @ a_weight.detach().to(device=work.device, dtype=torch.float64)
    )
    denominator = expected.norm().clamp_min(torch.finfo(torch.float64).eps)
    return float(((observed - expected).norm() / denominator).item())


def basis_energy_fraction(covariance: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of total activation energy lying in ``basis``."""

    cov = covariance.detach().to(dtype=torch.float64)
    directions = basis.detach().to(device=cov.device, dtype=torch.float64)
    numerator = torch.trace(directions.mT @ cov @ directions)
    denominator = torch.trace(cov).clamp_min(torch.finfo(torch.float64).eps)
    return float((numerator / denominator).item())
