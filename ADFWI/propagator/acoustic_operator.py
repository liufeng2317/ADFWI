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

from .acoustic_kernels import forward_kernel

ACOUSTIC_OPERATOR_STORAGE_MODES = frozenset({"device", "checkpoint", "none"})
ACOUSTIC_OPERATOR_BACKENDS = frozenset(
    {
        "compiled",
        "torch_reference",
        "custom_autograd_forward",
        "custom_autograd_remat",
    }
)


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
    checkpoint_segments: int = 1

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

        if not isinstance(self.checkpoint_segments, int) or self.checkpoint_segments <= 0:
            raise ValueError(
                "checkpoint_segments must be a positive integer, "
                f"got {self.checkpoint_segments!r}"
            )


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


def _torch_reference_pressure_operator(
    config: AcousticOperatorConfig,
    inputs: AcousticOperatorInputs,
) -> Tensor:
    """Run the current production kernel through the operator contract."""

    record = forward_kernel(
        config.nx,
        config.nz,
        config.dx,
        config.dz,
        config.nt,
        config.dt,
        config.nabc,
        config.free_surface,
        inputs.src_x,
        inputs.src_z,
        int(inputs.src_x.numel()),
        inputs.src_v,
        inputs.rcv_x,
        inputs.rcv_z,
        int(inputs.rcv_x.numel()),
        inputs.damp,
        inputs.vp,
        inputs.rho,
        checkpoint_segments=config.checkpoint_segments,
        save_forward_wavefield=False,
        pressure_only=True,
        device=inputs.vp.device,
        dtype=inputs.vp.dtype,
    )
    return record["p"]


class _AcousticPressureForwardOnlyFunction(torch.autograd.Function):
    """Forward-only custom-autograd shell for pressure receiver output.

    This is the first implementation step toward a real custom acoustic
    operator. Forward parity is tested through this shell before any backward
    implementation is added. Calling `.backward()` through this backend is
    intentionally rejected.
    """

    @staticmethod
    def forward(
        ctx,
        vp: Tensor,
        rho: Tensor,
        damp: Tensor,
        src_x: Tensor,
        src_z: Tensor,
        src_v: Tensor,
        rcv_x: Tensor,
        rcv_z: Tensor,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        nt: int,
        dt: float,
        nabc: int,
        free_surface: bool,
        checkpoint_segments: int,
    ) -> Tensor:
        del ctx
        record = forward_kernel(
            nx,
            nz,
            dx,
            dz,
            nt,
            dt,
            nabc,
            free_surface,
            src_x,
            src_z,
            int(src_x.numel()),
            src_v,
            rcv_x,
            rcv_z,
            int(rcv_x.numel()),
            damp,
            vp,
            rho,
            checkpoint_segments=checkpoint_segments,
            save_forward_wavefield=False,
            pressure_only=True,
            device=vp.device,
            dtype=vp.dtype,
        )
        return record["p"]

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError(
            "custom_autograd_forward only implements forward pressure parity; "
            "backward/vp gradient is not implemented yet"
        )


def _custom_autograd_forward_pressure_operator(
    config: AcousticOperatorConfig,
    inputs: AcousticOperatorInputs,
) -> Tensor:
    return _AcousticPressureForwardOnlyFunction.apply(
        inputs.vp,
        inputs.rho,
        inputs.damp,
        inputs.src_x,
        inputs.src_z,
        inputs.src_v,
        inputs.rcv_x,
        inputs.rcv_z,
        config.nx,
        config.nz,
        config.dx,
        config.dz,
        config.nt,
        config.dt,
        config.nabc,
        config.free_surface,
        config.checkpoint_segments,
    )


class _AcousticPressureRematFunction(torch.autograd.Function):
    """Custom-autograd shell that rematerializes forward in backward."""

    @staticmethod
    def forward(
        ctx,
        vp: Tensor,
        rho: Tensor,
        damp: Tensor,
        src_x: Tensor,
        src_z: Tensor,
        src_v: Tensor,
        rcv_x: Tensor,
        rcv_z: Tensor,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        nt: int,
        dt: float,
        nabc: int,
        free_surface: bool,
        checkpoint_segments: int,
    ) -> Tensor:
        ctx.save_for_backward(vp, rho, damp, src_x, src_z, src_v, rcv_x, rcv_z)
        ctx.config = (nx, nz, dx, dz, nt, dt, nabc, free_surface, checkpoint_segments)
        with torch.no_grad():
            record = forward_kernel(
                nx,
                nz,
                dx,
                dz,
                nt,
                dt,
                nabc,
                free_surface,
                src_x,
                src_z,
                int(src_x.numel()),
                src_v,
                rcv_x,
                rcv_z,
                int(rcv_x.numel()),
                damp,
                vp,
                rho,
                checkpoint_segments=checkpoint_segments,
                save_forward_wavefield=False,
                pressure_only=True,
                device=vp.device,
                dtype=vp.dtype,
            )
        return record["p"]

    @staticmethod
    def backward(ctx, grad_output):
        vp, rho, damp, src_x, src_z, src_v, rcv_x, rcv_z = ctx.saved_tensors
        nx, nz, dx, dz, nt, dt, nabc, free_surface, checkpoint_segments = ctx.config

        vp_req = vp.detach().requires_grad_(ctx.needs_input_grad[0])
        rho_req = rho.detach().requires_grad_(ctx.needs_input_grad[1])
        damp_req = damp.detach().requires_grad_(ctx.needs_input_grad[2])
        src_v_req = src_v.detach().requires_grad_(ctx.needs_input_grad[5])

        grad_targets = []
        target_positions = []
        for position, tensor in (
            (0, vp_req),
            (1, rho_req),
            (2, damp_req),
            (5, src_v_req),
        ):
            if tensor.requires_grad:
                grad_targets.append(tensor)
                target_positions.append(position)

        grads = [None] * 17
        if grad_targets:
            with torch.enable_grad():
                record = forward_kernel(
                    nx,
                    nz,
                    dx,
                    dz,
                    nt,
                    dt,
                    nabc,
                    free_surface,
                    src_x,
                    src_z,
                    int(src_x.numel()),
                    src_v_req,
                    rcv_x,
                    rcv_z,
                    int(rcv_x.numel()),
                    damp_req,
                    vp_req,
                    rho_req,
                    checkpoint_segments=checkpoint_segments,
                    save_forward_wavefield=False,
                    pressure_only=True,
                    device=vp.device,
                    dtype=vp.dtype,
                )
                computed_grads = torch.autograd.grad(
                    record["p"],
                    grad_targets,
                    grad_output,
                    allow_unused=True,
                )

            for position, grad in zip(target_positions, computed_grads):
                grads[position] = grad

        return tuple(grads)


def _custom_autograd_remat_pressure_operator(
    config: AcousticOperatorConfig,
    inputs: AcousticOperatorInputs,
) -> Tensor:
    return _AcousticPressureRematFunction.apply(
        inputs.vp,
        inputs.rho,
        inputs.damp,
        inputs.src_x,
        inputs.src_z,
        inputs.src_v,
        inputs.rcv_x,
        inputs.rcv_z,
        config.nx,
        config.nz,
        config.dx,
        config.dz,
        config.nt,
        config.dt,
        config.nabc,
        config.free_surface,
        config.checkpoint_segments,
    )


def acoustic_pressure_operator(
    config: AcousticOperatorConfig,
    inputs: AcousticOperatorInputs,
    *,
    backend: str = "compiled",
) -> Tensor:
    """Dispatch a pressure-only acoustic operator backend.

    `backend="torch_reference"` uses the current production kernel only to
    validate the operator contract and future compiled backend parity.
    `backend="custom_autograd_forward"` wraps the same forward-pressure formula
    in a custom autograd shell, but intentionally has no backward yet.
    `backend="custom_autograd_remat"` recomputes forward during backward to
    provide the first rematerialized gradient prototype.
    `backend="compiled"` is reserved for the future low-level implementation.
    """

    inputs.validate(config)
    if backend not in ACOUSTIC_OPERATOR_BACKENDS:
        valid = ", ".join(sorted(ACOUSTIC_OPERATOR_BACKENDS))
        raise ValueError(f"backend must be one of {{{valid}}}, got {backend!r}")

    if backend == "torch_reference":
        return _torch_reference_pressure_operator(config, inputs)
    if backend == "custom_autograd_forward":
        return _custom_autograd_forward_pressure_operator(config, inputs)
    if backend == "custom_autograd_remat":
        return _custom_autograd_remat_pressure_operator(config, inputs)

    raise CompiledAcousticOperatorUnavailable(
        "compiled acoustic pressure operator is not implemented yet; "
        "use backend='torch_reference' for contract parity checks or "
        "AcousticPropagator.forward for the production path"
    )
