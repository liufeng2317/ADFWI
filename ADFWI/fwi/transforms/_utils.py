"""Shared helpers for FWI data transforms."""

from __future__ import annotations

from typing import Any, Literal

import torch

from .base import Context, TensorPair


MaskTarget = Literal["synthetic", "observed", "both"]


def context_value(context: Context, key: str, required: bool = True) -> Any:
    if context is None or key not in context:
        if not required:
            return None
        raise ValueError(f"{key!r} must be provided either at construction time or in context")
    return context[key]


def validate_apply_to(apply_to: MaskTarget) -> MaskTarget:
    if apply_to not in {"synthetic", "observed", "both"}:
        raise ValueError("apply_to must be one of 'synthetic', 'observed', or 'both'")
    return apply_to


def apply_mask(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    apply_to: MaskTarget,
) -> TensorPair:
    if apply_to == "synthetic":
        return synthetic * mask, observed
    if apply_to == "observed":
        return synthetic, observed * mask
    return synthetic * mask, observed * mask


def as_mask_tensor(mask: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        result = mask.to(device=reference.device)
    else:
        result = torch.as_tensor(mask, device=reference.device)

    if result.dtype != torch.bool and reference.is_floating_point():
        result = result.to(dtype=reference.dtype)
    return result


def broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dim() == reference.dim() - 1 and mask.shape[0] == reference.shape[0] and mask.shape[-1] == reference.shape[-1]:
        mask = mask.unsqueeze(1)
    if mask.dim() > reference.dim():
        raise ValueError(f"mask has too many dimensions for data: mask={mask.shape}, data={reference.shape}")
    return mask
