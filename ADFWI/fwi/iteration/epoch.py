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
