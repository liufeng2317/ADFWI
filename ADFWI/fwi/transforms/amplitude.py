"""Amplitude transforms for waveform tensor pairs."""

from __future__ import annotations

import torch

from ADFWI.fwi.normalization import normalize_waveform

from .base import Context, DataTransform, TensorPair


class TraceNormalize(DataTransform):
    """Normalize each trace by its maximum absolute amplitude along time.

    The default assumes waveform tensors shaped ``[shot, time, receiver]`` and
    normalizes over ``dim=1``. All-zero traces remain zero.
    """

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        return normalize_waveform(data, dim=self.dim)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        return self._normalize(synthetic), self._normalize(observed)
