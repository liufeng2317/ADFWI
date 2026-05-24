"""Pure torch waveform transforms used by FWI data pipelines."""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base import Context, DataTransform, TensorPair


MaskTarget = Literal["synthetic", "observed", "both"]


def _context_value(context: Context, key: str, required: bool = True) -> Any:
    if context is None or key not in context:
        if not required:
            return None
        raise ValueError(f"{key!r} must be provided either at construction time or in context")
    return context[key]


def _validate_apply_to(apply_to: MaskTarget) -> MaskTarget:
    if apply_to not in {"synthetic", "observed", "both"}:
        raise ValueError("apply_to must be one of 'synthetic', 'observed', or 'both'")
    return apply_to


def _apply_mask(
    synthetic: torch.Tensor,
    observed: torch.Tensor,
    mask: torch.Tensor,
    apply_to: MaskTarget,
) -> TensorPair:
    if apply_to == "synthetic":
        return synthetic * mask, observed
    if apply_to == "observed":
        return synthetic, observed * mask
    return synthetic * mask, observed * mask


def _as_mask_tensor(mask: Any, reference: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, torch.Tensor):
        result = mask.to(device=reference.device)
    else:
        result = torch.as_tensor(mask, device=reference.device)

    if result.dtype != torch.bool and reference.is_floating_point():
        result = result.to(dtype=reference.dtype)
    return result


def _broadcast_mask(mask: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if mask.dim() == reference.dim() - 1 and mask.shape[0] == reference.shape[0] and mask.shape[-1] == reference.shape[-1]:
        mask = mask.unsqueeze(1)
    if mask.dim() > reference.dim():
        raise ValueError(f"mask has too many dimensions for data: mask={mask.shape}, data={reference.shape}")
    return mask


class TraceNormalize(DataTransform):
    """Normalize each trace by its maximum absolute amplitude along time.

    The default assumes waveform tensors shaped ``[shot, time, receiver]`` and
    normalizes over ``dim=1``. All-zero traces remain zero.
    """

    def __init__(self, dim: int = 1) -> None:
        self.dim = dim

    def _normalize(self, data: torch.Tensor) -> torch.Tensor:
        zero_trace = torch.sum(torch.abs(data), dim=self.dim, keepdim=True) == 0
        max_value = torch.max(torch.abs(data), dim=self.dim, keepdim=True).values
        max_value = max_value.masked_fill(zero_trace, 1)
        return data / max_value

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        return self._normalize(synthetic), self._normalize(observed)


class LowPassFilter(DataTransform):
    """Apply a differentiable torch low-pass FIR filter along the time axis.

    The default assumes waveform tensors shaped ``[shot, time, receiver]`` and
    filters over ``dim=1``. ``cutoff_freq`` is specified in Hz. Provide either
    ``dt`` or ``sampling_frequency`` at construction time, or pass them through
    ``context`` using the keys ``"dt"`` or ``"sampling_frequency"``.
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
    FWI workflows. It delegates to ``multiScaleProcessing.lpass`` and therefore
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

        from ADFWI.fwi.multiScaleProcessing import lpass

        return lpass(synthetic, observed, cutoff_freq, int(round(sampling_frequency)))


class LegacyOffsetMute(DataTransform):
    """Apply the legacy offset mute path through a transform.

    This preserves the existing ``ADFWI.utils.offset_mute.mute_offset`` behavior
    while making offset mute part of the reusable waveform preprocessing
    pipeline. It expects receiver geometry in context:
    ``receiver_mask``, ``src_x``, ``rcv_x``, and ``dx``.
    """

    def __init__(
        self,
        distance_threshold: Optional[float] = None,
        dx: Optional[float] = None,
        required: bool = True,
    ) -> None:
        self.distance_threshold = distance_threshold
        self.dx = dx
        self.required = required

    def _resolve_settings(self, context: Context) -> tuple[Optional[float], Optional[float]]:
        distance_threshold = self.distance_threshold
        if distance_threshold is None and context is not None:
            distance_threshold = context.get("offset_mute_threshold")

        dx = self.dx
        if dx is None and context is not None:
            dx = context.get("dx")

        if distance_threshold is None or dx is None:
            if not self.required:
                return None, None
            raise ValueError("offset_mute_threshold and dx must be provided")
        return float(distance_threshold), float(dx)

    def _selected_receiver_x(self, synthetic: torch.Tensor, context: Context) -> Optional[torch.Tensor]:
        receiver_mask = _context_value(context, "receiver_mask", required=self.required)
        src_x = _context_value(context, "src_x", required=self.required)
        rcv_x_list = _context_value(context, "rcv_x", required=self.required)

        if receiver_mask is None or src_x is None or rcv_x_list is None:
            if not self.required:
                return None
            raise ValueError("receiver_mask, src_x, and rcv_x must be provided")

        receiver_mask_2d = torch.as_tensor(receiver_mask).cpu()
        rcv_x_list = torch.as_tensor(rcv_x_list).cpu()
        rcv_x = torch.zeros(synthetic.shape[0], synthetic.shape[-1])
        for i in range(synthetic.shape[0]):
            active_receivers = np.argwhere(receiver_mask_2d[i].numpy()).reshape(-1).tolist()
            rcv_x[i] = rcv_x_list[active_receivers].squeeze()
        return rcv_x

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        distance_threshold, dx = self._resolve_settings(context)
        if distance_threshold is None or dx is None:
            return synthetic, observed

        rcv_x = self._selected_receiver_x(synthetic, context)
        if rcv_x is None:
            return synthetic, observed
        src_x = torch.as_tensor(_context_value(context, "src_x")).cpu()

        from ADFWI.utils.offset_mute import mute_offset

        return (
            mute_offset(rcv_x, src_x, dx, synthetic, distance_threshold),
            mute_offset(rcv_x, src_x, dx, observed, distance_threshold),
        )


class LegacyLateWindowMute(DataTransform):
    """Apply the legacy first-arrival late-window mute through a transform."""

    def __init__(
        self,
        late_window: Optional[float] = None,
        dt: Optional[float] = None,
        required: bool = True,
    ) -> None:
        self.late_window = late_window
        self.dt = dt
        self.required = required

    def _resolve_settings(self, context: Context) -> tuple[Optional[float], Optional[float]]:
        late_window = self.late_window
        if late_window is None and context is not None:
            late_window = context.get("late_window")

        dt = self.dt
        if dt is None and context is not None:
            dt = context.get("dt")

        if late_window is None or dt is None:
            if not self.required:
                return None, None
            raise ValueError("late_window and dt must be provided")
        return float(late_window), float(dt)

    def _mute(self, data: torch.Tensor, late_window: float, dt: float) -> torch.Tensor:
        from ADFWI.utils.first_arrivel_picking import apply_mute

        muted = data.clone()
        source = data.clone()
        for i in range(data.shape[0]):
            muted[i] = apply_mute(late_window, source[i], dt)
        return muted

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        late_window, dt = self._resolve_settings(context)
        if late_window is None or dt is None:
            return synthetic, observed
        return self._mute(synthetic, late_window, dt), self._mute(observed, late_window, dt)


class ReceiverMask(DataTransform):
    """Apply a receiver mask to waveform tensors.

    A 2-D mask shaped ``[shot, receiver]`` is automatically expanded to
    ``[shot, 1, receiver]`` for waveform tensors shaped ``[shot, time, receiver]``.
    If no mask is provided at construction time, ``context["receiver_mask"]`` is used.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = _validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else _context_value(context, "receiver_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = _broadcast_mask(_as_mask_tensor(raw_mask, synthetic), synthetic)
        return _apply_mask(synthetic, observed, mask, self.apply_to)


class DataMask(DataTransform):
    """Apply a sample-level data mask to waveform tensors.

    If no mask is provided at construction time, ``context["data_mask"]`` is used.
    The mask can be any shape broadcastable to the synthetic/observed tensors.
    """

    def __init__(self, mask: Optional[Any] = None, required: bool = True, apply_to: MaskTarget = "both") -> None:
        self.mask = mask
        self.required = required
        self.apply_to = _validate_apply_to(apply_to)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raw_mask = self.mask if self.mask is not None else _context_value(context, "data_mask", required=self.required)
        if raw_mask is None:
            return synthetic, observed
        mask = _broadcast_mask(_as_mask_tensor(raw_mask, synthetic), synthetic)
        return _apply_mask(synthetic, observed, mask, self.apply_to)
