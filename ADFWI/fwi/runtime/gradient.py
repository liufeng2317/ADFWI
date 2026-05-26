"""Runtime gradient-processing helpers shared by FWI drivers."""

from __future__ import annotations

from typing import Any, Optional, Type

import numpy as np
import torch

from ADFWI.utils import numpy2tensor

ACOUSTIC_PARAMETER_NAMES = ("vp", "rho")
ELASTIC_ISOTROPIC_PARAMETER_NAMES = ("vp", "vs", "rho")
ELASTIC_ANISOTROPIC_PARAMETER_NAMES = ("eps", "delta", "gamma")
ELASTIC_ANISOTROPIC_GRADIENT_NAMES = ("eps", "delta")


def parameter_specs(parameter_names, *, start_index=0):
    """Return ``(name, idx)`` pairs for legacy list-style gradient processors."""

    return [(name, start_index + idx) for idx, name in enumerate(parameter_names)]


def acoustic_parameter_names():
    """Return acoustic model parameters in the historical vp/rho order."""

    return list(ACOUSTIC_PARAMETER_NAMES)


def acoustic_gradient_parameter_specs():
    """Return acoustic gradient processor specs in the historical vp/rho order."""

    return parameter_specs(ACOUSTIC_PARAMETER_NAMES)


def elastic_parameter_names(*, include_anisotropic=False):
    """Return elastic model parameters used for regularization and snapshots."""

    names = list(ELASTIC_ISOTROPIC_PARAMETER_NAMES)
    if include_anisotropic:
        names.extend(ELASTIC_ANISOTROPIC_PARAMETER_NAMES)
    return names


def elastic_gradient_parameter_specs(*, include_anisotropic=False):
    """Return elastic gradient processor specs in the historical parameter order.

    The legacy elastic gradient processor path includes ``eps`` and ``delta``
    for anisotropic models. ``gamma`` is cached/regularized but is not passed to
    the gradient processor in the current ElasticFWI loop.
    """

    names = list(ELASTIC_ISOTROPIC_PARAMETER_NAMES)
    if include_anisotropic:
        names.extend(ELASTIC_ANISOTROPIC_GRADIENT_NAMES)
    return parameter_specs(names)


def process_parameter_gradient(
    parameter: torch.Tensor,
    gradient_processor: Any,
    *,
    model: Any,
    propagator: Any,
    forw: Any,
    idx: Optional[int] = None,
    processor_type: Optional[Type[Any]] = None,
) -> None:
    """Apply one legacy gradient processor to ``parameter.grad``.

    Parameters
    ----------
    parameter:
        Model parameter tensor whose ``.grad`` field has already been filled by
        autograd. The processed gradient is written back to this same field.
    gradient_processor:
        Either a single GradProcessor-like object or a sequence of such objects.
        A processor must provide ``forward(nz, nx, vmax, grad, forw)`` and return
        a NumPy gradient array.
    model:
        FWI model object providing ``nz`` and ``nx`` for processor geometry.
    propagator:
        Propagator object providing the target ``dtype`` and ``device`` for the
        restored PyTorch gradient tensor.
    forw:
        Forward wavefield array passed through to the legacy processor. Its
        exact shape depends on acoustic/elastic propagator output.
    idx:
        Parameter index used when ``gradient_processor`` is a list/sequence.
    processor_type:
        Optional class used to recognize the single-processor case without
        importing ``GradProcessor`` into this generic runtime module.

    Notes
    -----
    This helper intentionally keeps the historical CPU NumPy processor contract:
    gradients are converted to NumPy before processing, then converted back to
    ``propagator.dtype`` on ``propagator.device``.
    """
    with torch.no_grad():
        grads = parameter.grad.cpu().detach().numpy()
        vmax = np.max(parameter.cpu().detach().numpy())

        if processor_type is not None and isinstance(gradient_processor, processor_type):
            grads = gradient_processor.forward(
                nz=model.nz,
                nx=model.nx,
                vmax=vmax,
                grad=grads,
                forw=forw,
            )
        else:
            grads = gradient_processor[idx].forward(
                nz=model.nz,
                nx=model.nx,
                vmax=vmax,
                grad=grads,
                forw=forw,
            )

        parameter.grad = numpy2tensor(grads, dtype=propagator.dtype).to(propagator.device)


def process_named_parameter_gradients(model, parameter_specs, process_gradient_fn, *, forw):
    """Apply a driver's gradient processor to trainable named parameters.

    ``parameter_specs`` is an iterable of ``(name, idx)`` pairs. The FWI driver
    owns that physical parameter list; this helper only preserves the historical
    ``model.get_requires_grad(name)`` gate before calling ``process_gradient_fn``.
    """
    processed = []
    for name, idx in parameter_specs:
        if model.get_requires_grad(name):
            process_gradient_fn(getattr(model, name), forw=forw, idx=idx)
            processed.append(name)
    return processed
