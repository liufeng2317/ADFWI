"""One-batch propagator execution records for FWI drivers.

The helpers in this module only run the propagator for one selected shot batch
and return the propagator output together with the shot index used to select
observed data. Loss construction and wavefield interpretation are owned by
iteration and wavefield helpers.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ForwardBatchRecord:
    """Forward output and shot selection for one FWI batch."""

    shot_index: object
    record_waveform: object


def acoustic_forward_batch(
    propagator,
    batch_range,
    checkpoint_segments,
    *,
    save_forward_wavefield=True,
    storage_policy=None,
    pressure_only=False,
):
    """Run one acoustic forward batch and keep its shot selection with the record."""
    shot_index = batch_range.shot_index
    forward_kwargs = {
        "shot_index": shot_index,
        "checkpoint_segments": checkpoint_segments,
        "save_forward_wavefield": save_forward_wavefield,
        "pressure_only": pressure_only,
    }
    if storage_policy is not None:
        forward_kwargs["storage_policy"] = storage_policy
    record_waveform = propagator.forward(**forward_kwargs)
    return ForwardBatchRecord(shot_index=shot_index, record_waveform=record_waveform)


def elastic_forward_batch(propagator, batch_range, *, fd_order, checkpoint_segments):
    """Run one elastic forward batch and keep its shot selection with the record."""
    shot_index = batch_range.shot_index
    record_waveform = propagator.forward(
        fd_order=fd_order,
        shot_index=shot_index,
        checkpoint_segments=checkpoint_segments,
    )
    return ForwardBatchRecord(shot_index=shot_index, record_waveform=record_waveform)
