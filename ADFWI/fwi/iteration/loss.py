"""Batch loss construction and evaluation helpers for FWI iteration loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from ADFWI.fwi.misfit import Misfit, Misfit_NIM
from ADFWI.fwi.transforms import (
    DataMask,
    DataTransformPipeline,
    LegacyLateWindowMute,
    LegacyLowPassFilter,
    LegacyOffsetMute,
    TraceNormalize,
    select_or_mask_receivers,
)
from ADFWI.fwi.transforms.amplitude import normalize_waveform


ELASTIC_COMPONENTS = ("pressure", "vx", "vz")


@dataclass(frozen=True)
class LossInput:
    """One synthetic/observed waveform pair selected for a shot batch."""

    component: str
    synthetic: torch.Tensor
    observed: Any
    shot_index: Any
    weight: float = 1.0


@dataclass(frozen=True)
class ComponentLoss:
    """One evaluated component loss before and after weighting."""

    component: str
    loss: torch.Tensor
    weight: float
    weighted_loss: torch.Tensor


@dataclass(frozen=True)
class LossEvaluation:
    """Weighted data loss and per-component loss details."""

    data_loss: torch.Tensor
    component_losses: list[ComponentLoss]


@dataclass(frozen=True)
class BatchLoss:
    """Tensor loss and detached scalar value for epoch loss history."""

    tensor: Any
    scalar: float


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
    """Return active elastic component tensors and weights in stable order."""

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
    """Return validated elastic component weights for active components."""

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


def build_fwi_data_transform_pipeline(
    data_transform_pipeline: Optional[DataTransformPipeline],
    waveform_normalize: bool,
) -> tuple[DataTransformPipeline, bool]:
    """Build the legacy-compatible default FWI transform pipeline."""

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


def _shot_scoped_context_values(
    *,
    shot_index: Any,
    receiver_masks_2d: Any = None,
    src_x: Any = None,
    rcv_x: Any = None,
    data_masks: Optional[torch.Tensor] = None,
) -> dict[str, Any]:
    """Return transform context values that are meaningful only per shot batch."""

    if shot_index is None:
        return {}

    values: dict[str, Any] = {}
    if receiver_masks_2d is not None:
        values["receiver_mask"] = receiver_masks_2d[shot_index]
    if src_x is not None:
        values["src_x"] = src_x.cpu()[shot_index]
    if rcv_x is not None:
        values["rcv_x"] = rcv_x.cpu()
    if data_masks is not None:
        values["data_mask"] = data_masks[shot_index]
    return values


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
    """Build the metadata dictionary consumed by waveform transforms."""

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
    """Build the FWI transform context from inversion/propagator state."""

    shot_context = _shot_scoped_context_values(
        shot_index=shot_index,
        receiver_masks_2d=receiver_masks_2d,
        src_x=src_x,
        rcv_x=rcv_x,
        data_masks=data_masks,
    )

    return build_transform_context(
        shot_index=shot_index,
        cutoff_freq=cutoff_freq,
        dt=propagator_dt if propagator_dt is not None else default_dt,
        late_window=late_window,
        offset_mute_threshold=offset_mute_threshold,
        dx=dx,
        receiver_mask=shot_context.get("receiver_mask"),
        src_x=shot_context.get("src_x"),
        rcv_x=shot_context.get("rcv_x"),
        data_mask=shot_context.get("data_mask"),
    )


def prepare_loss_pair(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    receiver_mask: Any = None,
    data_transform_pipeline: Optional[DataTransformPipeline] = None,
    context: Optional[dict[str, Any]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare one synthetic/observed tensor pair before misfit evaluation."""

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
    """Build FWI transform context and prepare a loss pair."""

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


def sum_weighted_losses(losses: Sequence[torch.Tensor], *, device: Any = None) -> torch.Tensor:
    """Sum loss tensors while preserving autograd links."""

    if not losses:
        return torch.tensor(0.0, device=device)

    total = losses[0]
    for loss in losses[1:]:
        total = total + loss
    return total


def evaluate_misfit_loss(
    loss_fn: Any,
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    *,
    function_fallback: str = "call",
) -> torch.Tensor:
    """Evaluate a misfit while preserving legacy FWI calling conventions."""

    if isinstance(loss_fn, Misfit):
        return loss_fn.forward(synthetic, observed)
    if isinstance(loss_fn, Misfit_NIM):
        return loss_fn.apply(synthetic, observed, loss_fn.p, loss_fn.trans_type, loss_fn.theta)
    if function_fallback == "call":
        return loss_fn(synthetic, observed)
    if function_fallback == "apply":
        return loss_fn.apply(synthetic, observed)
    raise ValueError(f"unsupported misfit function fallback: {function_fallback}")


def evaluate_loss_inputs(
    loss_inputs: Sequence[LossInput],
    *,
    prepare_loss_pair: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Any,
    normalization: bool,
    function_fallback: str,
    cutoff_freq: Any = None,
    propagator_dt: Any = None,
    device: Any = None,
) -> LossEvaluation:
    """Prepare and evaluate FWI loss inputs with transparent component weights."""

    component_losses: list[ComponentLoss] = []
    weighted_losses: list[torch.Tensor] = []
    for loss_input in loss_inputs:
        synthetic, observed = prepare_loss_pair(
            loss_input.synthetic,
            loss_input.observed,
            shot_index=loss_input.shot_index,
            cutoff_freq=cutoff_freq,
            propagator_dt=propagator_dt,
        )
        if normalization:
            synthetic = normalize_waveform(synthetic)
            observed = normalize_waveform(observed)

        component_loss = evaluate_misfit_loss(
            loss_fn,
            synthetic,
            observed,
            function_fallback=function_fallback,
        )
        weighted_loss = component_loss * loss_input.weight
        weighted_losses.append(weighted_loss)
        component_losses.append(
            ComponentLoss(
                component=loss_input.component,
                loss=component_loss,
                weight=loss_input.weight,
                weighted_loss=weighted_loss,
            )
        )

    return LossEvaluation(
        data_loss=sum_weighted_losses(weighted_losses, device=device),
        component_losses=component_losses,
    )


def build_batch_loss(data_loss, regularization_loss=None) -> BatchLoss:
    """Combine data and optional regularization losses for one FWI batch.

    The scalar value intentionally mirrors the historical code path:
    ``data_loss.item()`` alone when no regularization is used, otherwise
    ``data_loss.item() + regularization_loss.item()``.
    """
    if regularization_loss is None:
        return BatchLoss(tensor=data_loss, scalar=data_loss.item())
    return BatchLoss(
        tensor=data_loss + regularization_loss,
        scalar=data_loss.item() + regularization_loss.item(),
    )
