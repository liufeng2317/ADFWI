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

from ADFWI.fwi.misfit import Misfit, Misfit_NIM
from ADFWI.fwi.normalization import normalize_waveform
from ADFWI.fwi.transforms import (
    DataMask,
    DataTransformPipeline,
    LegacyLateWindowMute,
    LegacyLowPassFilter,
    LegacyOffsetMute,
    TraceNormalize,
    select_or_mask_receivers,
)


def build_fwi_data_transform_pipeline(
    data_transform_pipeline: Optional[DataTransformPipeline],
    waveform_normalize: bool,
) -> tuple[DataTransformPipeline, bool]:
    """Build the legacy-compatible default FWI data transform pipeline.

    AcousticFWI and ElasticFWI share the same pre-loss transform order. When a
    custom pipeline is provided, it is appended after the legacy-compatible
    offset mute, late-window mute, low-pass filter, and synthetic data mask.
    """

    offset_mute = LegacyOffsetMute(required=False)
    late_mute = LegacyLateWindowMute(required=False)
    lowpass = LegacyLowPassFilter(required=False)
    data_mask = DataMask(required=False, apply_to="synthetic")
    transforms = [offset_mute, late_mute, lowpass, data_mask]

    if data_transform_pipeline is not None:
        return DataTransformPipeline(transforms + [data_transform_pipeline]), waveform_normalize

    if waveform_normalize:
        transforms.append(TraceNormalize())
        waveform_normalize = False
    return DataTransformPipeline(transforms), waveform_normalize


def evaluate_misfit_loss(
    loss_fn: Any,
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    function_fallback: str = "call",
) -> torch.Tensor:
    """Evaluate a misfit while preserving legacy FWI calling conventions.

    ``Misfit`` instances use ``forward``. ``Misfit_NIM`` keeps its custom
    autograd signature. Other loss functions either use ``loss_fn(...)`` or
    ``loss_fn.apply(...)`` depending on the historical caller.
    """

    if isinstance(loss_fn, Misfit):
        return loss_fn.forward(synthetic, observed)
    if isinstance(loss_fn, Misfit_NIM):
        return loss_fn.apply(synthetic, observed, loss_fn.p, loss_fn.trans_type, loss_fn.theta)
    if function_fallback == "call":
        return loss_fn(synthetic, observed)
    if function_fallback == "apply":
        return loss_fn.apply(synthetic, observed)
    raise ValueError(f"unsupported misfit function fallback: {function_fallback}")


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


def build_fwi_transform_context(
    *,
    shot_index: Any = None,
    cutoff_freq: Optional[float] = None,
    propagator_dt: Optional[float] = None,
    default_dt: Optional[float] = None,
    late_window: Optional[float] = None,
    offset_mute_threshold: Optional[float] = None,
    dx: Optional[float] = None,
    receiver_masks_2d: Any = None,
    src_x: Any = None,
    rcv_x: Any = None,
    data_masks: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Build the FWI transform context from inversion/propagator state.

    AcousticFWI and ElasticFWI use the same context fields before loss
    evaluation. This helper preserves the historical behavior: receiver masks,
    source x locations, receiver x locations, and data masks are only included
    when a shot index is available.
    """

    data_mask = None
    receiver_mask = None
    src_x_context = None
    rcv_x_context = None
    if shot_index is not None:
        if receiver_masks_2d is not None:
            receiver_mask = receiver_masks_2d[shot_index]
        if src_x is not None:
            src_x_context = src_x.cpu()[shot_index]
        if rcv_x is not None:
            rcv_x_context = rcv_x.cpu()
        if data_masks is not None:
            data_mask = data_masks[shot_index]

    return build_transform_context(
        shot_index=shot_index,
        cutoff_freq=cutoff_freq,
        dt=propagator_dt if propagator_dt is not None else default_dt,
        late_window=late_window,
        offset_mute_threshold=offset_mute_threshold,
        dx=dx,
        receiver_mask=receiver_mask,
        src_x=src_x_context,
        rcv_x=rcv_x_context,
        data_mask=data_mask,
    )


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


def prepare_fwi_loss_pair(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    shot_index: Any = None,
    cutoff_freq: Optional[float] = None,
    propagator_dt: Optional[float] = None,
    default_dt: Optional[float] = None,
    late_window: Optional[float] = None,
    offset_mute_threshold: Optional[float] = None,
    dx: Optional[float] = None,
    receiver_masks_2d: Any = None,
    src_x: Any = None,
    rcv_x: Any = None,
    data_masks: Optional[torch.Tensor] = None,
    data_transform_pipeline: Optional[DataTransformPipeline] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build FWI transform context and prepare a loss pair.

    This is the shared AcousticFWI/ElasticFWI pre-loss path. It keeps receiver
    selection before the transform pipeline and uses the same context values as
    ``build_fwi_transform_context``.
    """

    context = build_fwi_transform_context(
        shot_index=shot_index,
        cutoff_freq=cutoff_freq,
        propagator_dt=propagator_dt,
        default_dt=default_dt,
        late_window=late_window,
        offset_mute_threshold=offset_mute_threshold,
        dx=dx,
        receiver_masks_2d=receiver_masks_2d,
        src_x=src_x,
        rcv_x=rcv_x,
        data_masks=data_masks,
    )
    return prepare_loss_pair(
        synthetic,
        observed,
        receiver_mask=context.get("receiver_mask"),
        data_transform_pipeline=data_transform_pipeline,
        context=context,
    )


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
