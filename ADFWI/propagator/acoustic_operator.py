"""Acoustic operator boundary for future compiled/autograd kernels.

This module defines the narrow contract for the next acoustic performance
path. It does not replace `acoustic_kernels.py` and is not used by
`AcousticPropagator` by default. The purpose is to keep the future compiled
operator interface explicit before any low-level kernel implementation is
added.
"""

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


ACOUSTIC_OPERATOR_STORAGE_MODES = frozenset({"device", "checkpoint", "none"})


class CompiledAcousticOperatorUnavailable(RuntimeError):
    """Raised when the compiled acoustic operator backend is requested."""


def _format_expected_shape(shape: Iterable[int]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def _require_shape(name: str, tensor: Tensor, expected_shape: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have shape {_format_expected_shape(expected_shape)}, "
            f"got {tuple(tensor.shape)}"
        )


def _require_integer_tensor(name: str, tensor: Tensor) -> None:
    if tensor.dtype not in (torch.int32, torch.int64, torch.long):
        raise TypeError(f"{name} must use an integer dtype, got {tensor.dtype}")


@dataclass(frozen=True)
class AcousticOperatorConfig:
    """Non-differentiable configuration for a pressure-only acoustic operator."""

    nx: int
    nz: int
    dx: float
    dz: float
    nt: int
    dt: float
    nabc: int
    free_surface: bool
    storage_mode: str = "checkpoint"
    pressure_only: bool = True

    def validate(self) -> None:
        """Validate scalar metadata before dispatching to a custom operator."""

        for name in ("nx", "nz", "nt"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if not isinstance(self.nabc, int) or self.nabc < 0:
            raise ValueError(f"nabc must be a non-negative integer, got {self.nabc!r}")

        for name in ("dx", "dz", "dt"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        if self.storage_mode not in ACOUSTIC_OPERATOR_STORAGE_MODES:
            valid = ", ".join(sorted(ACOUSTIC_OPERATOR_STORAGE_MODES))
            raise ValueError(f"storage_mode must be one of {{{valid}}}, got {self.storage_mode!r}")

        if not self.pressure_only:
            raise ValueError("the first acoustic operator contract only supports pressure_only=True")


@dataclass(frozen=True)
class AcousticOperatorInputs:
    """Tensor inputs for the first pressure-only acoustic operator target."""

    src_x: Tensor
    src_z: Tensor
    src_v: Tensor
    rcv_x: Tensor
    rcv_z: Tensor
    damp: Tensor
    vp: Tensor
    rho: Tensor

    def validate(self, config: AcousticOperatorConfig) -> None:
        """Validate tensor shapes, dtypes, and device consistency."""

        config.validate()

        if self.src_x.ndim != 1:
            raise ValueError(f"src_x must be 1-D, got {self.src_x.ndim}-D")
        if self.src_z.ndim != 1:
            raise ValueError(f"src_z must be 1-D, got {self.src_z.ndim}-D")
        if self.rcv_x.ndim != 1:
            raise ValueError(f"rcv_x must be 1-D, got {self.rcv_x.ndim}-D")
        if self.rcv_z.ndim != 1:
            raise ValueError(f"rcv_z must be 1-D, got {self.rcv_z.ndim}-D")

        src_n = int(self.src_x.numel())
        rcv_n = int(self.rcv_x.numel())
        _require_shape("src_z", self.src_z, (src_n,))
        _require_shape("rcv_z", self.rcv_z, (rcv_n,))
        _require_shape("src_v", self.src_v, (src_n, config.nt))
        _require_shape("vp", self.vp, (config.nz, config.nx))
        _require_shape("rho", self.rho, (config.nz, config.nx))
        _require_shape(
            "damp",
            self.damp,
            (config.nz + 2 * config.nabc, config.nx + 2 * config.nabc),
        )

        for name in ("src_x", "src_z", "rcv_x", "rcv_z"):
            _require_integer_tensor(name, getattr(self, name))

        reference_device = self.vp.device
        for name in ("src_x", "src_z", "src_v", "rcv_x", "rcv_z", "damp", "rho"):
            tensor = getattr(self, name)
            if tensor.device != reference_device:
                raise ValueError(
                    f"{name} must be on device {reference_device}, got {tensor.device}"
                )


def compiled_acoustic_operator_available() -> bool:
    """Return whether a compiled acoustic operator backend is available."""

    return False


def acoustic_pressure_operator(config: AcousticOperatorConfig, inputs: AcousticOperatorInputs) -> Tensor:
    """Dispatch the future compiled pressure-only acoustic operator.

    The function validates the public contract now, then raises an explicit
    availability error until a compiled forward/backward backend is added.
    """

    inputs.validate(config)
    raise CompiledAcousticOperatorUnavailable(
        "compiled acoustic pressure operator is not implemented yet; "
        "use AcousticPropagator.forward for the production path"
    )
