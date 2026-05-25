"""Waveform normalization helpers shared by FWI data paths and transforms."""

from __future__ import annotations

import torch


def normalize_waveform(data: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Normalize each waveform trace by its maximum absolute amplitude.

    All-zero traces keep a denominator of 1 so they remain zero and do not
    introduce NaNs.
    """

    zero_trace = torch.sum(torch.abs(data), dim=dim, keepdim=True) == 0
    max_value = torch.max(torch.abs(data), dim=dim, keepdim=True).values
    max_value = max_value.masked_fill(zero_trace, 1)
    return data / max_value
