"""Synthetic/observed waveform pair preparation before FWI misfit evaluation."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ADFWI.fwi.transforms import DataTransformPipeline, select_or_mask_receivers


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
    """Build the metadata dictionary consumed by waveform transforms.

    Key meanings follow the FWI data path: ``shot_index`` selects the current
    source batch; ``cutoff_freq`` and ``dt`` parameterize low-pass filtering;
    ``late_window`` controls first-arrival muting; ``offset_mute_threshold`` and
    ``dx`` define offset muting; ``receiver_mask`` and ``data_mask`` describe
    trace/data availability. Optional geometry arrays ``src_x`` and ``rcv_x``
    are included only when needed by offset-based transforms.
    """

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
    """Prepare one synthetic/observed tensor pair before misfit evaluation.

    ``synthetic`` and ``observed`` are waveform tensors, normally shaped
    ``[shot, time, receiver]`` after shot selection. Receiver selection runs
    first because trace-missing data can change the receiver dimension. The
    transform pipeline then operates only on same-shape tensor pairs.
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
