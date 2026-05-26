"""Loss dispatch and accumulation helpers for FWI data paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch

from ADFWI.fwi.misfit import Misfit, Misfit_NIM
from ADFWI.fwi.normalization import normalize_waveform

from .inputs import LossInput


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


def sum_weighted_losses(losses: Sequence[torch.Tensor], *, device: Any = None) -> torch.Tensor:
    """Sum loss tensors while preserving autograd links.

    Empty component sets are valid for edge-case tests and return a scalar zero
    on the requested device. Non-empty inputs start from the first loss tensor so
    dtype, device, and autograd history are inherited from real loss values.
    """

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
    """Prepare and evaluate FWI loss inputs with transparent component weights.

    The caller still owns the physical choice of ``loss_inputs``. This helper
    only applies the shared pre-loss preparation, optional legacy waveform
    normalization, misfit dispatch, and component-weight summation.
    """

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
