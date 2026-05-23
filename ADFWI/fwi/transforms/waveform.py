"""Pure torch waveform transforms used by FWI data pipelines."""

from __future__ import annotations

from typing import Any, Literal, Optional

import torch

from .base import Context, DataTransform, TensorPair


MaskTarget = Literal["synthetic", "observed", "both"]


def _context_value(context: Context, key: str, required: bool = True) -> Any:
    if context is None or key not in context:
        if not required:
            return None
        raise ValueError(f"{key!r} must be provided either at construction time or in context")
    return context[key]


def _validate_apply_to(apply_to: MaskTarget) -> MaskTarget:
    if apply_to not in {"synthetic", "observed", "both"}:
        raise ValueError("apply_to must be one of 'synthetic', 'observed', or 'both'")
    return apply_to


def _apply_mask(
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


def _as_mask_tensor(mask: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        result = mask.to(device=reference.device)
    else:
        result = torch.as_tensor(mask, device=reference.device)

    if result.dtype != torch.bool and reference.is_floating_point():
        result = result.to(dtype=reference.dtype)
    return result


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dim() == reference.dim() - 1 and mask.shape[0] == reference.shape[0] and mask.shape[-1] == reference.shape[-1]:
        mask = mask.unsqueeze(1)
    if mask.dim() > reference.dim():
        raise ValueError(f"mask has too many dimensions for data: mask={mask.shape}, data={reference.shape}")
    return mask


class TraceNormalize(DataTransform):
    """Normalize each trace by its maximum absolute amplitude along time.

    The default assumes waveform tensors shaped ``[shot, time, receiver]`` and
    normalizes over ``dim=1``. All-zero traces remain zero.
    """

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        zero_trace = torch.sum(torch.abs(data), dim=self.dim, keepdim=True) == 0
        max_value = torch.max(torch.abs(data), dim=self.dim, keepdim=True).values
        max_value = max_value.masked_fill(zero_trace, 1)
        return data / max_value

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        return self._normalize(synthetic), self._normalize(observed)


class ReceiverMask(DataTransform):
    """Apply a receiver mask to waveform tensors.

    A 2-D mask shaped ``[shot, receiver]`` is automatically expanded to
    ``[shot, 1, receiver]`` for waveform tensors shaped ``[shot, time, receiver]``.
    If no mask is provided at construction time, ``context["receiver_mask"]`` is used.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = _validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else _context_value(context, "receiver_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = _broadcast_mask(_as_mask_tensor(raw_mask, synthetic), synthetic)
        return _apply_mask(synthetic, observed, mask, self.apply_to)


class DataMask(DataTransform):
    """Apply a sample-level data mask to waveform tensors.

    If no mask is provided at construction time, ``context["data_mask"]`` is used.
    The mask can be any shape broadcastable to the synthetic/observed tensors.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = _validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else _context_value(context, "data_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = _broadcast_mask(_as_mask_tensor(raw_mask, synthetic), synthetic)
        return _apply_mask(synthetic, observed, mask, self.apply_to)
