"""Batch loss bookkeeping for FWI iteration loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BatchLoss:
    """Tensor loss and detached scalar value for epoch loss history."""

    tensor: Any
    scalar: float


def build_batch_loss(data_loss, regularization_loss=None) -> BatchLoss:
    """Combine data and optional regularization losses for one FWI batch.

    The scalar value intentionally mirrors the historical code path:
    ``data_loss.item()`` alone when no regularization is used, otherwise
    ``data_loss.item() + regularization_loss.item()``.
    """
    if regularization_loss is None:
        return BatchLoss(tensor=data_loss, scalar=data_loss.item())
    return BatchLoss(
        tensor=data_loss + regularization_loss,
        scalar=data_loss.item() + regularization_loss.item(),
    )
