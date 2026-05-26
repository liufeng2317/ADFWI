"""Runtime helpers for FWI driver construction and execution."""

from .backend import align_regularization_backend, validate_model_propagator_devices
from .cache import (
    append_epoch_loss,
    append_model_snapshots,
    append_required_gradient_snapshots,
    should_cache_epoch,
    snapshot_model_parameters,
    tensor_to_numpy,
)
from .gradient import process_named_parameter_gradients, process_parameter_gradient
from .regularization import calculate_regularization_loss
from .wavefield import (
    acoustic_pressure_waveforms,
    accumulate_named_wavefields,
    accumulate_wavefield,
    elastic_gradient_wavefields,
    select_elastic_gradient_wavefield,
    wavefield_to_numpy,
)

__all__ = [
    "accumulate_named_wavefields",
    "accumulate_wavefield",
    "acoustic_pressure_waveforms",
    "align_regularization_backend",
    "elastic_gradient_wavefields",
    "append_epoch_loss",
    "append_model_snapshots",
    "append_required_gradient_snapshots",
    "calculate_regularization_loss",
    "process_named_parameter_gradients",
    "process_parameter_gradient",
    "select_elastic_gradient_wavefield",
    "should_cache_epoch",
    "snapshot_model_parameters",
    "tensor_to_numpy",
    "validate_model_propagator_devices",
    "wavefield_to_numpy",
]
