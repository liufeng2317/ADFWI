"""Construction-time backend alignment helpers for FWI drivers.

These helpers validate or align objects that are created before an inversion
loop starts. They do not choose a backend and they do not move model or
propagator tensors during iteration.
"""

from __future__ import annotations

from typing import Any

import torch


def validate_model_propagator_devices(model: Any, propagator: Any) -> None:
    """Fail early when a model and propagator use different devices.

    ``model`` and ``propagator`` are expected to expose a ``device`` attribute.
    Matching devices are required because the propagator reads differentiable
    model tensors directly during forward modeling.
    """

    if model.device != propagator.device:
        raise ValueError(
            f"Model device {model.device} and propagator device {propagator.device} are inconsistent. "
            "Create them with the same backend or configure ADFWI.backends before constructing them."
        )


def align_regularization_backend(regularization_fn: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move a regularization object onto the FWI backend.

    Parameters
    ----------
    regularization_fn:
        Regularization object used by AcousticFWI/ElasticFWI, or ``None``. The
        object is updated in place because existing regularization classes keep
        ``device``/``dtype`` metadata and may store tensor masks or weights.
    device:
        Target ``torch.device`` shared by model and propagator.
    dtype:
        Floating-point dtype used for model-space regularization arithmetic.

    Floating and complex tensor attributes follow ``dtype``. Integer and boolean
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
