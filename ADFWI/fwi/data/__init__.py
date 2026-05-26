"""FWI data contract helpers before misfit evaluation.

This package is the FWI-layer data contract: it aligns synthetic and observed
waveform pairs before they enter a misfit. It may orchestrate receiver
selection, transform context construction, loss dispatch, and elastic component
bookkeeping. Concrete waveform operations stay in ``ADFWI.fwi.transforms``;
misfit formulas stay in ``ADFWI.fwi.misfit``; propagator execution stays in
``ADFWI.propagator``.

The package keeps the historical ``ADFWI.fwi.data`` import surface stable while
organizing implementation by responsibility.
"""

from ADFWI.fwi.normalization import normalize_waveform

from .components import (
    ELASTIC_COMPONENTS,
    elastic_component_loss_inputs,
    elastic_observed_components,
    elastic_pressure,
    elastic_synthetic_components,
    normalize_elastic_component_weights,
)
from .inputs import LossInput, acoustic_pressure_loss_input, elastic_loss_inputs
from .loss import ComponentLoss, LossEvaluation, evaluate_loss_inputs, evaluate_misfit_loss, sum_weighted_losses
from .pipeline import build_fwi_data_transform_pipeline
from .preparation import (
    build_fwi_transform_context,
    build_transform_context,
    prepare_fwi_loss_pair,
    prepare_loss_pair,
)

__all__ = [
    "ELASTIC_COMPONENTS",
    "ComponentLoss",
    "LossInput",
    "LossEvaluation",
    "build_fwi_data_transform_pipeline",
    "build_fwi_transform_context",
    "build_transform_context",
    "acoustic_pressure_loss_input",
    "elastic_component_loss_inputs",
    "elastic_loss_inputs",
    "elastic_observed_components",
    "elastic_pressure",
    "elastic_synthetic_components",
    "evaluate_loss_inputs",
    "evaluate_misfit_loss",
    "normalize_elastic_component_weights",
    "normalize_waveform",
    "prepare_fwi_loss_pair",
    "prepare_loss_pair",
    "sum_weighted_losses",
]
