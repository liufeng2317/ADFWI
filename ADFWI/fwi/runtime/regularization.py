"""Runtime regularization helpers shared by FWI drivers."""

from __future__ import annotations

import torch


def calculate_regularization_loss(model_param, weight_x, weight_z, regularization_fn):
    """Calculate regularization loss for one model parameter.

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
