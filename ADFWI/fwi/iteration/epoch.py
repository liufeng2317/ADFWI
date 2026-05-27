"""Epoch-level optimizer update helpers for FWI loops."""


def apply_epoch_update_step(optimizer, scheduler, model, *, closure=None):
    """Apply the legacy optimizer/scheduler/model-constraint order.

    The FWI drivers still own gradient computation and any optimizer-specific
    closure construction. This helper only preserves the shared epoch tail:
    optimizer step first, scheduler step second, model constraint last.
    """
    if closure is None:
        step_result = optimizer.step()
    else:
        step_result = optimizer.step(closure=closure)
    scheduler.step()
    model.forward()
    return step_result


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
