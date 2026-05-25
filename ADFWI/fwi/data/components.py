"""Elastic component bookkeeping for FWI data preparation."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch


ELASTIC_COMPONENTS = ("pressure", "vx", "vz")


def elastic_pressure(txx: torch.Tensor, tzz: torch.Tensor) -> torch.Tensor:
    """Return the pressure component used by elastic FWI."""

    return -(txx + tzz)


def elastic_synthetic_components(record_waveform: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Build named elastic synthetic components from propagator output."""

    return {
        "pressure": elastic_pressure(record_waveform["txx"], record_waveform["tzz"]),
        "vx": record_waveform["vx"],
        "vz": record_waveform["vz"],
    }


def elastic_observed_components(data: dict[str, Any]) -> dict[str, Any]:
    """Build named elastic observed components from seismic data arrays."""

    return {
        "pressure": -(data["txx"] + data["tzz"]),
        "vx": data["vx"],
        "vz": data["vz"],
    }


def elastic_component_loss_inputs(
    synthetic_components: Mapping[str, torch.Tensor],
    observed_components: Mapping[str, Any],
    inversion_components: Sequence[str],
    component_weights: Mapping[str, float],
) -> list[tuple[str, torch.Tensor, Any, float]]:
    """Return active elastic component tensors and weights in stable order.

    ElasticFWI evaluates components in ``ELASTIC_COMPONENTS`` order regardless
    of user input order. This helper keeps the component selection and weight
    lookup outside the inversion loop while leaving loss evaluation in ElasticFWI.
    """

    active_components = set(inversion_components)
    inputs: list[tuple[str, torch.Tensor, Any, float]] = []
    for component in ELASTIC_COMPONENTS:
        if component not in active_components:
            continue
        inputs.append((
            component,
            synthetic_components[component],
            observed_components[component],
            component_weights[component],
        ))
    return inputs


def normalize_elastic_component_weights(
    inversion_components: Sequence[str],
    component_weights: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Return validated elastic component weights for active components.

    Missing active components default to weight 1.0. Unknown component names are
    rejected so typos do not silently change inversion behavior.
    """

    active_components = tuple(inversion_components)
    unknown_active = set(active_components) - set(ELASTIC_COMPONENTS)
    if unknown_active:
        raise ValueError(f"unsupported elastic inversion components: {sorted(unknown_active)}")

    if component_weights is None:
        component_weights = {}
    unknown_weights = set(component_weights) - set(ELASTIC_COMPONENTS)
    if unknown_weights:
        raise ValueError(f"unsupported elastic component weights: {sorted(unknown_weights)}")

    weights: dict[str, float] = {}
    for component in active_components:
        weight = float(component_weights.get(component, 1.0))
        if weight < 0.0:
            raise ValueError(f"elastic component weight must be non-negative: {component}={weight}")
        weights[component] = weight
    return weights
