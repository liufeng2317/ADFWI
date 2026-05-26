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
from .forward import ForwardBatchRecord, acoustic_forward_batch, elastic_forward_batch
from .gradient import (
    acoustic_gradient_parameter_specs,
    acoustic_parameter_names,
    elastic_gradient_parameter_specs,
    elastic_parameter_names,
    parameter_specs,
    process_named_parameter_gradients,
    process_parameter_gradient,
)
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
    "acoustic_gradient_parameter_specs",
    "acoustic_parameter_names",
    "align_regularization_backend",
    "elastic_gradient_wavefields",
    "elastic_gradient_parameter_specs",
    "elastic_parameter_names",
    "ForwardBatchRecord",
    "acoustic_forward_batch",
    "elastic_forward_batch",
    "append_epoch_loss",
    "append_model_snapshots",
    "append_required_gradient_snapshots",
    "calculate_regularization_loss",
    "parameter_specs",
    "process_named_parameter_gradients",
    "process_parameter_gradient",
    "select_elastic_gradient_wavefield",
    "should_cache_epoch",
    "snapshot_model_parameters",
    "tensor_to_numpy",
    "validate_model_propagator_devices",
    "wavefield_to_numpy",
]
