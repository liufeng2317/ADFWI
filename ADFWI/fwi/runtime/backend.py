"""Runtime backend helpers shared by FWI drivers."""

from __future__ import annotations

from typing import Any

import torch


def validate_model_propagator_devices(model: Any, propagator: Any) -> None:
    """Fail early when a model and propagator are on different devices."""

    if model.device != propagator.device:
        raise ValueError(
            f"Model device {model.device} and propagator device {propagator.device} are inconsistent. "
            "Create them with the same backend or configure ADFWI.backends before constructing them."
        )


def align_regularization_backend(regularization_fn: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move regularization metadata and tensor attributes to an FWI backend.

    Floating and complex tensors follow the FWI dtype. Integer and boolean
    tensors keep their dtype while moving device. ``None`` is accepted so FWI
    constructors can call this unconditionally.
    """

    if regularization_fn is None:
        return None

    regularization_fn.device = device
    regularization_fn.dtype = dtype
    for name, value in vars(regularization_fn).items():
        if not torch.is_tensor(value):
            continue
        to_kwargs = {"device": device}
        if value.is_floating_point() or value.is_complex():
            to_kwargs["dtype"] = dtype
        setattr(regularization_fn, name, value.to(**to_kwargs))
    return regularization_fn
