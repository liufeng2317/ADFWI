"""Receiver selection helpers for FWI observed/synthetic matching."""

from __future__ import annotations

from typing import Any

import torch

from ._utils import as_mask_tensor, broadcast_mask


def select_or_mask_receivers(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    receiver_mask: Any,
) -> torch.Tensor:
    """Match synthetic receiver traces to observed data using a receiver mask.

    If synthetic and observed already share the same shape, this preserves the
    legacy partial-data behavior by multiplying synthetic data by the broadcast
    receiver mask. If observed data has fewer receiver traces, active synthetic
    receiver traces are gathered per shot in receiver-mask order.
    """

    receiver_mask_2d = torch.as_tensor(receiver_mask, device=synthetic.device).bool()
    if receiver_mask_2d.dim() != 2:
        raise ValueError(f"receiver_mask must be 2-D [shot, receiver], got {tuple(receiver_mask_2d.shape)}")
    if receiver_mask_2d.shape[0] != synthetic.shape[0]:
        raise ValueError(
            "receiver_mask shot dimension must match synthetic data, "
            f"got {receiver_mask_2d.shape[0]} and {synthetic.shape[0]}"
        )

    if synthetic.shape == observed.shape:
        mask = broadcast_mask(as_mask_tensor(receiver_mask_2d, synthetic), synthetic)
        return synthetic * mask

    selected = torch.zeros_like(observed, device=synthetic.device)
    for shot in range(synthetic.shape[0]):
        active_receivers = torch.nonzero(receiver_mask_2d[shot], as_tuple=False).flatten()
        if active_receivers.numel() != observed.shape[-1]:
            raise ValueError(
                "active receiver count must match observed receiver dimension "
                f"for shot {shot}, got {active_receivers.numel()} and {observed.shape[-1]}"
            )
        selected[shot] = synthetic[shot].index_select(-1, active_receivers)
    return selected
