"""Small helpers for FWI iteration loops.

These helpers keep batch slicing, loss bookkeeping, and progress labels
identical between acoustic and elastic FWI without moving numerical work out of
the model-specific loops.
"""

from .loss import BatchLoss, build_batch_loss
from .progress import set_batch_description
from .range import BatchRange, iter_batch_ranges

__all__ = [
    "BatchLoss",
    "BatchRange",
    "build_batch_loss",
    "iter_batch_ranges",
    "set_batch_description",
]
