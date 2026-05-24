"""Mask transforms for same-shape waveform tensor pairs."""

from __future__ import annotations

from typing import Any, Optional

import torch

from .base import Context, DataTransform, TensorPair
from ._utils import MaskTarget, apply_mask, as_mask_tensor, broadcast_mask, context_value, validate_apply_to


class ReceiverMask(DataTransform):
    """Apply a receiver mask to waveform tensors.

    A 2-D mask shaped ``[shot, receiver]`` is automatically expanded to
    ``[shot, 1, receiver]`` for waveform tensors shaped ``[shot, time, receiver]``.
    If no mask is provided at construction time, ``context["receiver_mask"]`` is used.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else context_value(context, "receiver_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = broadcast_mask(as_mask_tensor(raw_mask, synthetic), synthetic)
        return apply_mask(synthetic, observed, mask, self.apply_to)


class DataMask(DataTransform):
    """Apply a sample-level data mask to waveform tensors.

    If no mask is provided at construction time, ``context["data_mask"]`` is used.
    The mask can be any shape broadcastable to the synthetic/observed tensors.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else context_value(context, "data_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = broadcast_mask(as_mask_tensor(raw_mask, synthetic), synthetic)
        return apply_mask(synthetic, observed, mask, self.apply_to)
