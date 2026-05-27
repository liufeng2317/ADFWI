"""Public FWI data-contract facade before misfit evaluation.

This package is the FWI-layer data contract: it aligns synthetic and observed
waveform pairs before they enter a misfit. It may orchestrate receiver
selection, transform context construction, loss dispatch, and elastic component
bookkeeping. Concrete waveform operations stay in ``ADFWI.fwi.transforms``;
misfit formulas stay in ``ADFWI.fwi.misfit``; propagator execution stays in
``ADFWI.propagator``.

Use ``ADFWI.fwi.data`` as the stable user-facing import surface for data
contract helpers. Framework internals should import from owner modules such as
``ADFWI.fwi.data.preparation`` or ``ADFWI.fwi.data.loss`` to keep dependencies
explicit.
"""

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
    # Records and constants
    "ELASTIC_COMPONENTS",
    "ComponentLoss",
    "LossInput",
    "LossEvaluation",
    # Pipeline and pair preparation
    "build_fwi_data_transform_pipeline",
    "build_fwi_transform_context",
    "build_transform_context",
    "prepare_fwi_loss_pair",
    "prepare_loss_pair",
    # Loss inputs and evaluation
    "acoustic_pressure_loss_input",
    "elastic_component_loss_inputs",
    "elastic_loss_inputs",
    "evaluate_loss_inputs",
    "evaluate_misfit_loss",
    "sum_weighted_losses",
    # Elastic component helpers
    "elastic_observed_components",
    "elastic_pressure",
    "elastic_synthetic_components",
    "normalize_elastic_component_weights",
]
