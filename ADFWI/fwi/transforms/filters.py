"""Low-pass filtering transforms for waveform tensor pairs.

``LowPassFilter`` is the differentiable torch FIR path intended for device-native
pipelines. ``LegacyLowPassFilter`` delegates to the historical multiscale
implementation when exact legacy behavior is required.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import Context, DataTransform, TensorPair


class LowPassFilter(DataTransform):
    """Apply a differentiable torch low-pass FIR filter along the time axis.

    The default assumes waveform tensors shaped ``[shot, time, receiver]`` and
    filters over ``dim=1``. ``cutoff_freq`` is specified in Hz. Provide either
    ``dt`` or ``sampling_frequency`` at construction time, or pass them through
    ``context`` using the keys ``"dt"`` or ``"sampling_frequency"``.

    This path is intended for torch-native CPU/CUDA/NPU execution. Its response
    is not bitwise identical to the legacy SciPy/autograd low-pass path.
    """

    def __init__(
        self,
        cutoff_freq: Optional[float] = None,
        dt: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        filter_length: int = 31,
        dim: int = 1,
        required: bool = True,
    ) -> None:
        if dt is not None and sampling_frequency is not None:
            raise ValueError("provide either dt or sampling_frequency, not both")
        if filter_length < 3:
            raise ValueError("filter_length must be at least 3")
        if filter_length % 2 == 0:
            filter_length += 1
        self.cutoff_freq = cutoff_freq
        self.dt = dt
        self.sampling_frequency = sampling_frequency
        self.filter_length = filter_length
        self.dim = dim
        self.required = required

    def _resolve_settings(self, context: Context) -> tuple[Optional[float], Optional[float]]:
        cutoff_freq = self.cutoff_freq
        if cutoff_freq is None and context is not None:
            cutoff_freq = context.get("cutoff_freq")

        sampling_frequency = self.sampling_frequency
        if sampling_frequency is None:
            dt = self.dt
            if dt is None and context is not None:
                dt = context.get("dt")
            if dt is not None:
                sampling_frequency = 1.0 / float(dt)
        if sampling_frequency is None and context is not None:
            sampling_frequency = context.get("sampling_frequency")

        if cutoff_freq is None or sampling_frequency is None:
            if not self.required:
                return None, None
            raise ValueError("cutoff_freq and dt or sampling_frequency must be provided")
        return float(cutoff_freq), float(sampling_frequency)

    def _kernel(self, reference: torch.Tensor, cutoff_freq: float, sampling_frequency: float) -> torch.Tensor:
        if cutoff_freq <= 0:
            raise ValueError("cutoff_freq must be positive")
        nyquist = 0.5 * sampling_frequency
        if cutoff_freq >= nyquist:
            return torch.ones(1, 1, 1, device=reference.device, dtype=reference.dtype)

        cutoff = cutoff_freq / sampling_frequency
        center = self.filter_length // 2
        n = torch.arange(self.filter_length, device=reference.device, dtype=reference.dtype) - center
        kernel = 2.0 * cutoff * torch.sinc(2.0 * cutoff * n)
        window = torch.hann_window(self.filter_length, periodic=False, device=reference.device, dtype=reference.dtype)
        kernel = kernel * window
        kernel = kernel / kernel.sum()
        return kernel.reshape(1, 1, -1)

    def _filter(self, data: torch.Tensor, cutoff_freq: float, sampling_frequency: float) -> torch.Tensor:
        dim = self.dim if self.dim >= 0 else data.dim() + self.dim
        if dim < 0 or dim >= data.dim():
            raise ValueError(f"dim {self.dim} is out of bounds for data with {data.dim()} dimensions")

        kernel = self._kernel(data, cutoff_freq, sampling_frequency)
        if kernel.shape[-1] == 1:
            return data

        moved = data.movedim(dim, -1)
        original_shape = moved.shape
        traces = moved.reshape(-1, 1, original_shape[-1])
        filtered = F.conv1d(traces, kernel, padding=kernel.shape[-1] // 2)
        filtered = filtered.reshape(original_shape).movedim(-1, dim)
        return filtered.to(dtype=data.dtype)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        cutoff_freq, sampling_frequency = self._resolve_settings(context)
        if cutoff_freq is None or sampling_frequency is None:
            return synthetic, observed
        return (
            self._filter(synthetic, cutoff_freq, sampling_frequency),
            self._filter(observed, cutoff_freq, sampling_frequency),
        )


class LegacyLowPassFilter(DataTransform):
    """Apply the legacy SciPy/autograd low-pass path through a transform.

    This transform is intended for precision-preserving migration of existing
    FWI workflows. It delegates to ``ADFWI.fwi.multiscale.lpass`` and therefore
    matches the legacy numerical behavior, including its custom backward path,
    but it is not a pure torch/NPU-native implementation.
    """

    def __init__(
        self,
        cutoff_freq: Optional[float] = None,
        dt: Optional[float] = None,
        sampling_frequency: Optional[float] = None,
        required: bool = True,
    ) -> None:
        if dt is not None and sampling_frequency is not None:
            raise ValueError("provide either dt or sampling_frequency, not both")
        self.cutoff_freq = cutoff_freq
        self.dt = dt
        self.sampling_frequency = sampling_frequency
        self.required = required

    def _resolve_settings(self, context: Context) -> tuple[Optional[float], Optional[float]]:
        cutoff_freq = self.cutoff_freq
        if cutoff_freq is None and context is not None:
            cutoff_freq = context.get("cutoff_freq")

        sampling_frequency = self.sampling_frequency
        if sampling_frequency is None:
            dt = self.dt
            if dt is None and context is not None:
                dt = context.get("dt")
            if dt is not None:
                sampling_frequency = 1.0 / float(dt)
        if sampling_frequency is None and context is not None:
            sampling_frequency = context.get("sampling_frequency")

        if cutoff_freq is None or sampling_frequency is None:
            if not self.required:
                return None, None
            raise ValueError("cutoff_freq and dt or sampling_frequency must be provided")
        return float(cutoff_freq), float(sampling_frequency)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        cutoff_freq, sampling_frequency = self._resolve_settings(context)
        if cutoff_freq is None or sampling_frequency is None:
            return synthetic, observed

        from ADFWI.fwi.multiscale import lpass

        return lpass(synthetic, observed, cutoff_freq, int(round(sampling_frequency)))
