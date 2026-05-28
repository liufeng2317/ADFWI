"""Model regularization loss helpers shared by FWI drivers.

These helpers implement the common summation mechanics for model-space
regularization. The FWI drivers still choose the ordered physical parameter
list and the corresponding x/z weights.
"""

from __future__ import annotations

import torch


def calculate_regularization_loss(model_param, weight_x, weight_z, regularization_fn):
    """Calculate regularization loss for one model parameter tensor.

    ``model_param`` is usually one differentiable field such as ``vp``, ``vs``,
    ``rho``, ``eps``, ``delta``, or ``gamma``. ``weight_x`` and ``weight_z`` are
    the regularization strengths in horizontal and vertical directions.

    This preserves the historical AcousticFWI/ElasticFWI behavior: parameters
    that do not require gradients contribute a scalar zero on their device;
    active parameters set the regularization object's x/z weights before calling
    its ``forward`` method when either weight is positive.
    """

    regularization_loss = torch.tensor(0.0, device=model_param.device)
    if model_param.requires_grad:
        regularization_fn.alphax = weight_x
        regularization_fn.alphaz = weight_z
        if regularization_fn.alphax > 0 or regularization_fn.alphaz > 0:
            regularization_loss = regularization_fn.forward(model_param)
    return regularization_loss


def calculate_model_regularization_loss(model, parameter_names, weights_x, weights_z, regularization_fn):
    """Sum regularization losses for an ordered list of model parameters.

    ``parameter_names`` owns the physical parameter order selected by the FWI
    driver. The weight lists are indexed with the same order to preserve the
    historical AcousticFWI and ElasticFWI behavior.
    """

    regularization_loss = None
    for idx, name in enumerate(parameter_names):
        parameter_loss = calculate_regularization_loss(
            getattr(model, name),
            weights_x[idx],
            weights_z[idx],
            regularization_fn,
        )
        regularization_loss = parameter_loss if regularization_loss is None else regularization_loss + parameter_loss
    return regularization_loss
