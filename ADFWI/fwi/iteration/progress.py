"""Progress-bar helpers for FWI iteration loops."""

from __future__ import annotations

from .range import BatchRange


def set_batch_description(progress_bar, batch_range: BatchRange, batch_count: int) -> None:
    """Set a legacy single-batch shot range description on a tqdm bar.

    The historical FWI loops only displayed the shot range when there was a
    single batch. Keeping that rule here preserves progress output while moving
    iteration bookkeeping out of acoustic/elastic code.
    """
    if batch_count == 1:
        progress_bar.set_description(f"Shot:{batch_range.begin} to {batch_range.end}")


def finalize_epoch_progress(
    progress_bar,
    *,
    epoch_id,
    loss_epoch,
    cache_result=False,
    cache_callback=None,
):
    """Apply legacy epoch-final cache and progress-label behavior.

    The FWI driver supplies ``cache_callback`` so parameter choices, gradient
    choices, and figure output remain model-specific.
    """
    if cache_result and cache_callback is not None:
        cache_callback(epoch_id=epoch_id, loss_epoch=loss_epoch)
    progress_bar.set_description("Iter:{},Loss:{:.4}".format(epoch_id + 1, loss_epoch))
