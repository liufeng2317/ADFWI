"""Loss dispatch and accumulation helpers for FWI data paths."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from ADFWI.fwi.misfit import Misfit, Misfit_NIM


def sum_weighted_losses(losses: Sequence[torch.Tensor], *, device: Any = None) -> torch.Tensor:
    """Sum loss tensors while preserving autograd links.

    Empty component sets are valid for edge-case tests and return a scalar zero
    on the requested device. Non-empty inputs start from the first loss tensor so
    dtype, device, and autograd history are inherited from real loss values.
    """

    if not losses:
        return torch.tensor(0.0, device=device)

    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total


def evaluate_misfit_loss(
    loss_fn: Any,
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    function_fallback: str = "call",
) -> torch.Tensor:
    """Evaluate a misfit while preserving legacy FWI calling conventions.

    ``Misfit`` instances use ``forward``. ``Misfit_NIM`` keeps its custom
    autograd signature. Other loss functions either use ``loss_fn(...)`` or
    ``loss_fn.apply(...)`` depending on the historical caller.
    """

    if isinstance(loss_fn, Misfit):
        return loss_fn.forward(synthetic, observed)
    if isinstance(loss_fn, Misfit_NIM):
        return loss_fn.apply(synthetic, observed, loss_fn.p, loss_fn.trans_type, loss_fn.theta)
    if function_fallback == "call":
        return loss_fn(synthetic, observed)
    if function_fallback == "apply":
        return loss_fn.apply(synthetic, observed)
    raise ValueError(f"unsupported misfit function fallback: {function_fallback}")
