"""Lightweight conversion helpers for legacy examples and framework glue.

These helpers intentionally preserve historical ADFWI behavior. They are not a
general backend policy layer; device selection and dtype policy live elsewhere.
"""

import numpy as np
import torch


def numpy2tensor(value, dtype=torch.float32):
    """Convert non-tensor input to a detached tensor with the requested dtype.

    Existing tensors are returned unchanged, including their dtype, device, and
    ``requires_grad`` flag. This preserves legacy call sites that pass already
    prepared tensors through this helper.
    """
    if not torch.is_tensor(value):
        return torch.tensor(value, requires_grad=False, dtype=dtype)
    return value


def tensor2numpy(value):
    """Detach a CPU tensor to NumPy, or return non-tensor input unchanged."""
    if not torch.is_tensor(value):
        return value
    return value.detach().numpy()


def gpu2cpu(value):
    """Move a tensor to CPU NumPy form, or return non-tensor input unchanged."""
    if torch.is_tensor(value):
        if value.requires_grad:
            if value.device == 'cpu':
                value = value.detach().numpy()
            else:
                value = value.cpu().detach().numpy()
        else:
            if value.device == 'cpu':
                return value.numpy()
            else:
                return value.cpu().numpy()
    return value


def list2numpy(value):
    """Convert Python lists to NumPy arrays, or return other inputs unchanged."""
    if isinstance(value, list):
        return np.array(value)
    return value


def numpy2list(value):
    """Convert array-like non-list input with ``tolist()``, or return lists unchanged."""
    if not isinstance(value, list):
        return value.tolist()
    return value
