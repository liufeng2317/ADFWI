"""Receiver selection helpers for FWI observed/synthetic matching."""

from __future__ import annotations

from typing import Any

import numpy as np
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

    receiver_mask_2d = torch.as_tensor(receiver_mask).cpu()
    if synthetic.shape == observed.shape:
        mask = broadcast_mask(as_mask_tensor(receiver_mask_2d, synthetic), synthetic)
        return synthetic * mask

    selected = torch.zeros_like(observed, device=synthetic.device)
    for shot in range(synthetic.shape[0]):
        active_receivers = np.argwhere(receiver_mask_2d[shot].numpy()).reshape(-1).tolist()
        traces = synthetic[shot, ..., active_receivers]
        if traces.dim() == selected[shot].dim() - 1:
            traces = traces.unsqueeze(-1)
        selected[shot] = traces
    return selected
