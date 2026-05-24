"""FWI data preparation helpers before misfit evaluation.

This module is the FWI-layer data contract: it aligns synthetic and observed
waveform pairs before they enter a misfit. It may orchestrate receiver
selection, transform context construction, and component bookkeeping. Concrete
waveform operations stay in ``ADFWI.fwi.transforms``; misfit formulas stay in
``ADFWI.fwi.misfit``; propagator execution stays in ``ADFWI.propagator``.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

from ADFWI.fwi.transforms import DataTransformPipeline, select_or_mask_receivers


def build_transform_context(
    *,
    shot_index: Any = None,
    cutoff_freq: Optional[float] = None,
    dt: Optional[float] = None,
    late_window: Optional[float] = None,
    offset_mute_threshold: Optional[float] = None,
    dx: Optional[float] = None,
    receiver_mask: Any = None,
    src_x: Any = None,
    rcv_x: Any = None,
    data_mask: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Build the context consumed by FWI data transform pipelines."""

    context: dict[str, Any] = {
        "shot_index": shot_index,
        "cutoff_freq": cutoff_freq,
        "dt": dt,
        "late_window": late_window,
        "offset_mute_threshold": offset_mute_threshold,
        "dx": dx,
    }
    if receiver_mask is not None:
        context["receiver_mask"] = receiver_mask
    if src_x is not None:
        context["src_x"] = src_x
    if rcv_x is not None:
        context["rcv_x"] = rcv_x
    if data_mask is not None:
        context["data_mask"] = data_mask
    return context


def prepare_loss_pair(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    receiver_mask: Any = None,
    data_transform_pipeline: Optional[DataTransformPipeline] = None,
    context: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare a synthetic/observed pair for same-shape loss evaluation.

    Receiver selection runs first because trace-missing data can change the
    receiver dimension. The regular transform pipeline then operates only on
    same-shape synthetic/observed tensors.
    """

    if receiver_mask is not None:
        synthetic = select_or_mask_receivers(synthetic, observed, receiver_mask)

    if data_transform_pipeline is not None:
        synthetic, observed = data_transform_pipeline(synthetic, observed, context=context)

    return synthetic, observed


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
