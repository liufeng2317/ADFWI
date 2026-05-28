"""Legacy mute wrappers for waveform tensor pairs.

These classes expose existing offset and first-arrival mute utilities through
the transform pipeline interface. They preserve historical behavior; they are
not redesigned torch-native mute operators.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .base import Context, DataTransform, TensorPair
from ._utils import context_value


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
        receiver_mask = context_value(context, "receiver_mask", required=self.required)
        src_x = context_value(context, "src_x", required=self.required)
        rcv_x_list = context_value(context, "rcv_x", required=self.required)

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
        src_x = torch.as_tensor(context_value(context, "src_x")).cpu()

        from ADFWI.utils.offset_mute import mute_offset

        return (
            mute_offset(rcv_x, src_x, dx, synthetic, distance_threshold),
            mute_offset(rcv_x, src_x, dx, observed, distance_threshold),
        )


class LegacyLateWindowMute(DataTransform):
    """Apply the legacy first-arrival late-window mute through a transform.

    The implementation calls ``ADFWI.utils.first_arrivel_picking.apply_mute``
    per shot to preserve the previous mute behavior.
    """

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
