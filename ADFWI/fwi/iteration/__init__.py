"""Small helpers for FWI iteration loops.

These helpers keep batch slicing, loss bookkeeping, and progress labels
identical between acoustic and elastic FWI without moving numerical work out of
the model-specific loops.
"""

from .epoch import apply_epoch_update_step
from .loss import (
    AcousticBatchStepResult,
    BatchLoss,
    ElasticBatchStepResult,
    apply_acoustic_batch_loss_step,
    apply_batch_loss_step,
    apply_elastic_batch_loss_step,
    build_batch_loss,
)
from .progress import finalize_epoch_progress, set_batch_description
from .range import BatchRange, iter_batch_ranges

__all__ = [
    "AcousticBatchStepResult",
    "BatchLoss",
    "BatchRange",
    "ElasticBatchStepResult",
    "apply_acoustic_batch_loss_step",
    "apply_elastic_batch_loss_step",
    "apply_epoch_update_step",
    "apply_batch_loss_step",
    "build_batch_loss",
    "finalize_epoch_progress",
    "iter_batch_ranges",
    "set_batch_description",
]
