"""One-batch forward/loss/backward steps for FWI iteration loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ADFWI.fwi.runtime.forward import acoustic_forward_batch, elastic_forward_batch
from ADFWI.fwi.runtime.wavefield import (
    acoustic_pressure_waveforms,
    accumulate_named_wavefields,
    accumulate_wavefield,
    elastic_gradient_wavefields,
)

from .batches import set_batch_description
from .loss import (
    acoustic_pressure_loss_input,
    build_batch_loss,
    elastic_loss_inputs,
    evaluate_loss_inputs,
)


@dataclass(frozen=True)
class AcousticBatchStepResult:
    """Updated acoustic epoch loss and accumulated gradient wavefield."""

    epoch_loss_scalar: Any
    accumulated_wavefield: Any


@dataclass(frozen=True)
class ElasticBatchStepResult:
    """Updated elastic epoch loss and accumulated gradient wavefields."""

    epoch_loss_scalar: Any
    accumulated_wavefields: Any


def apply_batch_loss_step(
    epoch_loss_scalar,
    data_loss,
    regularization_loss=None,
    *,
    progress_bar=None,
    batch_range=None,
    batch_count=None,
):
    """Apply the shared loss/backward/progress step for one FWI batch."""

    batch_loss = build_batch_loss(data_loss, regularization_loss)
    epoch_loss_scalar = epoch_loss_scalar + batch_loss.scalar
    batch_loss.tensor.backward()
    if progress_bar is not None and batch_range is not None and batch_count is not None:
        set_batch_description(progress_bar, batch_range, batch_count)
    return epoch_loss_scalar


def apply_acoustic_batch_loss_step(
    *,
    epoch_loss_scalar,
    accumulated_wavefield,
    propagator,
    batch_range,
    checkpoint_segments,
    observed_pressure,
    prepare_loss_pair,
    loss_fn,
    normalization,
    cutoff_freq,
    regularization_loss_fn=None,
    progress_bar=None,
    batch_count=None,
    device=None,
) -> AcousticBatchStepResult:
    """Run one acoustic FWI batch and apply the shared loss/backward step."""

    forward_batch = acoustic_forward_batch(propagator, batch_range, checkpoint_segments)
    loss_input = acoustic_pressure_loss_input(
        forward_batch.record_waveform,
        observed_pressure,
        forward_batch.shot_index,
    )
    _, forward_wavefield_p = acoustic_pressure_waveforms(forward_batch.record_waveform)
    accumulated_wavefield = accumulate_wavefield(accumulated_wavefield, forward_wavefield_p)

    loss_evaluation = evaluate_loss_inputs(
        [loss_input],
        prepare_loss_pair=prepare_loss_pair,
        loss_fn=loss_fn,
        normalization=normalization,
        function_fallback="apply",
        cutoff_freq=cutoff_freq,
        propagator_dt=propagator.dt,
        device=device,
    )
    regularization_loss = regularization_loss_fn() if regularization_loss_fn is not None else None
    epoch_loss_scalar = apply_batch_loss_step(
        epoch_loss_scalar,
        loss_evaluation.data_loss,
        regularization_loss,
        progress_bar=progress_bar,
        batch_range=batch_range,
        batch_count=batch_count,
    )
    return AcousticBatchStepResult(
        epoch_loss_scalar=epoch_loss_scalar,
        accumulated_wavefield=accumulated_wavefield,
    )


def apply_elastic_batch_loss_step(
    *,
    epoch_loss_scalar,
    accumulated_wavefields,
    propagator,
    batch_range,
    fd_order,
    checkpoint_segments,
    observed_components,
    inversion_components,
    component_weights,
    prepare_loss_pair,
    loss_fn,
    normalization,
    cutoff_freq,
    regularization_loss_fn=None,
    progress_bar=None,
    batch_count=None,
    device=None,
) -> ElasticBatchStepResult:
    """Run one elastic FWI batch and apply the shared loss/backward step."""

    forward_batch = elastic_forward_batch(
        propagator,
        batch_range,
        fd_order=fd_order,
        checkpoint_segments=checkpoint_segments,
    )
    batch_wavefields = elastic_gradient_wavefields(
        forward_batch.record_waveform,
        inversion_components,
    )
    accumulate_named_wavefields(accumulated_wavefields, batch_wavefields)

    loss_evaluation = evaluate_loss_inputs(
        elastic_loss_inputs(
            forward_batch.record_waveform,
            observed_components,
            inversion_components,
            component_weights,
            forward_batch.shot_index,
        ),
        prepare_loss_pair=prepare_loss_pair,
        loss_fn=loss_fn,
        normalization=normalization,
        function_fallback="call",
        cutoff_freq=cutoff_freq,
        propagator_dt=propagator.dt,
        device=device,
    )
    regularization_loss = regularization_loss_fn() if regularization_loss_fn is not None else None
    epoch_loss_scalar = apply_batch_loss_step(
        epoch_loss_scalar,
        loss_evaluation.data_loss,
        regularization_loss,
        progress_bar=progress_bar,
        batch_range=batch_range,
        batch_count=batch_count,
    )
    return ElasticBatchStepResult(
        epoch_loss_scalar=epoch_loss_scalar,
        accumulated_wavefields=accumulated_wavefields,
    )
