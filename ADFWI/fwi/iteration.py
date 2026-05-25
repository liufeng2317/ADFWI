'''
Small helpers for FWI iteration loops.

These helpers keep batch slicing identical between acoustic and elastic FWI
without moving any numerical work out of the model-specific loops.
'''
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class BatchRange:
    """A contiguous shot batch used by FWI forward/backward loops."""

    batch: int
    begin: int
    end: int
    shot_index: np.ndarray


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


def iter_batch_ranges(n_shots: int, batch_size: Optional[int] = None) -> Iterator[BatchRange]:
    """Yield contiguous shot batches for an FWI epoch.

    ``batch_size=None`` and ``batch_size > n_shots`` both mean full-batch mode,
    matching the historical AcousticFWI/ElasticFWI behavior.
    """
    if n_shots <= 0:
        raise ValueError("n_shots must be positive")
    if batch_size is None or batch_size > n_shots:
        batch_size = n_shots
    if batch_size <= 0:
        raise ValueError("batch_size must be positive or None")

    for batch, begin in enumerate(range(0, n_shots, batch_size)):
        end = min(begin + batch_size, n_shots)
        yield BatchRange(batch=batch, begin=begin, end=end, shot_index=np.arange(begin, end))
