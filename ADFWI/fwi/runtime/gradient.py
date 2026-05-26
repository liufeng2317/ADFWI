"""Runtime gradient-processing helpers shared by FWI drivers."""

from __future__ import annotations

import numpy as np
import torch

from ADFWI.utils import numpy2tensor


def process_parameter_gradient(
    parameter,
    gradient_processor,
    *,
    model,
    propagator,
    forw,
    idx=None,
    processor_type=None,
):
    """Apply the legacy gradient processor and restore grad on propagator backend."""
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
