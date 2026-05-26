"""Raw FWI loss-input records before transform preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from .components import elastic_component_loss_inputs, elastic_synthetic_components


@dataclass(frozen=True)
class LossInput:
    """One synthetic/observed waveform pair selected for a shot batch."""

    component: str
    synthetic: torch.Tensor
    observed: Any
    shot_index: Any
    weight: float = 1.0


def acoustic_pressure_loss_input(
    record_waveform: Mapping[str, torch.Tensor],
    observed_pressure: Any,
    shot_index: Any,
) -> LossInput:
    """Return the acoustic pressure loss input for one forward batch."""

    return LossInput(
        component="pressure",
        synthetic=record_waveform["p"],
        observed=observed_pressure[shot_index],
        shot_index=shot_index,
    )


def elastic_loss_inputs(
    record_waveform: Mapping[str, torch.Tensor],
    observed_components: Mapping[str, Any],
    inversion_components: Sequence[str],
    component_weights: Mapping[str, float],
    shot_index: Any,
) -> list[LossInput]:
    """Return active elastic component loss inputs for one forward batch."""

    synthetic_components = elastic_synthetic_components(record_waveform)
    inputs: list[LossInput] = []
    for component, synthetic, observed_component, weight in elastic_component_loss_inputs(
        synthetic_components,
        observed_components,
        inversion_components,
        component_weights,
    ):
        inputs.append(
            LossInput(
                component=component,
                synthetic=synthetic,
                observed=observed_component[shot_index],
                shot_index=shot_index,
                weight=weight,
            )
        )
    return inputs
