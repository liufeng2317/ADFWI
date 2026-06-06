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

from .acoustic_kernels import forward_kernel, step_forward_pressure_only

ACOUSTIC_OPERATOR_STORAGE_MODES = frozenset({"device", "checkpoint", "none"})
ACOUSTIC_OPERATOR_BACKENDS = frozenset(
    {
        "compiled",
        "torch_reference",
        "custom_autograd_forward",
        "custom_autograd_remat",
    }
)
ACOUSTIC_SEGMENT_BACKENDS = frozenset(
    {"torch_reference", "custom_autograd_forward", "compiled"}
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


@dataclass(frozen=True)
class AcousticPressureSegmentConfig:
    """Non-differentiable metadata for a pressure-only time segment.

    This config is intentionally narrower than ``AcousticOperatorConfig``: it
    targets one local time segment with already-built PML coefficients and
    incoming wavefield state. It is not used by production propagators unless a
    caller opts in explicitly.
    """

    nx: int
    nz: int
    dx: float
    dz: float
    dt: float
    nabc: int
    free_surface: bool

    def validate(self) -> None:
        for name in ("nx", "nz"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")

        if not isinstance(self.nabc, int) or self.nabc < 0:
            raise ValueError(f"nabc must be a non-negative integer, got {self.nabc!r}")

        for name in ("dx", "dz", "dt"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass(frozen=True)
class AcousticPressureSegmentInputs:
    """Tensor inputs for one pressure-only local time segment."""

    src_x: Tensor
    src_z: Tensor
    src_index: Tensor
    src_v: Tensor
    rcv_x: Tensor
    rcv_z: Tensor
    kappa1: Tensor
    alpha1: Tensor
    kappa2: Tensor
    alpha2: Tensor
    kappa3: Tensor
    p: Tensor
    u: Tensor
    w: Tensor

    def validate(self, config: AcousticPressureSegmentConfig) -> None:
        config.validate()

        if self.src_x.ndim != 1:
            raise ValueError(f"src_x must be 1-D, got {self.src_x.ndim}-D")
        if self.src_z.ndim != 1:
            raise ValueError(f"src_z must be 1-D, got {self.src_z.ndim}-D")
        if self.src_index.ndim != 1:
            raise ValueError(f"src_index must be 1-D, got {self.src_index.ndim}-D")
        if self.rcv_x.ndim != 1:
            raise ValueError(f"rcv_x must be 1-D, got {self.rcv_x.ndim}-D")
        if self.rcv_z.ndim != 1:
            raise ValueError(f"rcv_z must be 1-D, got {self.rcv_z.ndim}-D")
        if self.src_v.ndim != 2:
            raise ValueError(f"src_v must have shape (src_n, segment_nt), got {tuple(self.src_v.shape)}")

        src_n = int(self.src_x.numel())
        rcv_n = int(self.rcv_x.numel())
        segment_nt = int(self.src_v.shape[1])
        nx_pml = config.nx + 2 * config.nabc
        nz_pml = config.nz + 2 * config.nabc

        _require_shape("src_z", self.src_z, (src_n,))
        _require_shape("src_index", self.src_index, (src_n,))
        _require_shape("src_v", self.src_v, (src_n, segment_nt))
        _require_shape("rcv_z", self.rcv_z, (rcv_n,))
        _require_shape("kappa1", self.kappa1, (nz_pml, nx_pml))
        _require_shape("alpha1", self.alpha1, (nz_pml, nx_pml))
        _require_shape("kappa2", self.kappa2, (nz_pml, nx_pml - 1))
        _require_shape("alpha2", self.alpha2, (nz_pml, nx_pml - 1))
        _require_shape("kappa3", self.kappa3, (nz_pml - 1, nx_pml))
        _require_shape("p", self.p, (src_n, nz_pml, nx_pml))
        _require_shape("u", self.u, (src_n, nz_pml, nx_pml - 1))
        _require_shape("w", self.w, (src_n, nz_pml - 1, nx_pml))

        for name in ("src_x", "src_z", "src_index", "rcv_x", "rcv_z"):
            _require_integer_tensor(name, getattr(self, name))

        reference_device = self.p.device
        for name in (
            "src_x",
            "src_z",
            "src_index",
            "src_v",
            "rcv_x",
            "rcv_z",
            "kappa1",
            "alpha1",
            "kappa2",
            "alpha2",
            "kappa3",
            "u",
            "w",
        ):
            tensor = getattr(self, name)
            if tensor.device != reference_device:
                raise ValueError(
                    f"{name} must be on device {reference_device}, got {tensor.device}"
                )


@dataclass(frozen=True)
class AcousticPressureSegmentOutput:
    """Output state from one pressure-only local time segment."""

    p: Tensor
    u: Tensor
    w: Tensor
    rcv_p: Tensor


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
    if config.checkpoint_segments != 1:
        raise ValueError(
            "custom_autograd_remat currently supports checkpoint_segments=1 only; "
            "production reentrant checkpointing is incompatible with the "
            "torch.autograd.grad rematerialized backward prototype"
        )
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


def acoustic_pressure_segment(
    config: AcousticPressureSegmentConfig,
    inputs: AcousticPressureSegmentInputs,
    *,
    backend: str = "torch_reference",
) -> AcousticPressureSegmentOutput:
    """Run one opt-in pressure-only acoustic time segment.

    This is an experimental boundary for future fused time-segment work. The
    reference backend delegates to the current production single-segment
    function and returns the final wavefield state plus local receiver samples.
    """

    inputs.validate(config)
    if backend not in ACOUSTIC_SEGMENT_BACKENDS:
        valid = ", ".join(sorted(ACOUSTIC_SEGMENT_BACKENDS))
        raise ValueError(f"backend must be one of {{{valid}}}, got {backend!r}")

    if backend == "custom_autograd_forward":
        p, u, w, rcv_p = _AcousticPressureSegmentForwardOnlyFunction.apply(
            inputs.p,
            inputs.u,
            inputs.w,
            inputs.src_v,
            inputs.src_x,
            inputs.src_z,
            inputs.src_index,
            inputs.rcv_x,
            inputs.rcv_z,
            inputs.kappa1,
            inputs.alpha1,
            inputs.kappa2,
            inputs.alpha2,
            inputs.kappa3,
            config.nx,
            config.nz,
            config.dx,
            config.dz,
            config.dt,
            config.nabc,
            config.free_surface,
        )
        return AcousticPressureSegmentOutput(p=p, u=u, w=w, rcv_p=rcv_p)

    if backend != "torch_reference":
        raise CompiledAcousticOperatorUnavailable(
            "compiled acoustic pressure segment is not implemented yet; "
            "use backend='torch_reference' for segment contract checks"
        )

    p, u, w, rcv_p, _ = step_forward_pressure_only(
        config.nx,
        config.nz,
        config.dx,
        config.dz,
        config.dt,
        config.nabc,
        config.free_surface,
        inputs.src_x,
        inputs.src_z,
        int(inputs.src_x.numel()),
        inputs.src_index,
        inputs.src_v,
        inputs.rcv_x,
        inputs.rcv_z,
        int(inputs.rcv_x.numel()),
        inputs.kappa1,
        inputs.alpha1,
        inputs.kappa2,
        inputs.alpha2,
        inputs.kappa3,
        9.0 / 8.0,
        -1.0 / 24.0,
        inputs.p,
        inputs.u,
        inputs.w,
        save_forward_wavefield=False,
        accumulate_wavefield_in_grad=False,
        device=inputs.p.device,
        dtype=inputs.p.dtype,
    )
    return AcousticPressureSegmentOutput(p=p, u=u, w=w, rcv_p=rcv_p)


class _AcousticPressureSegmentForwardOnlyFunction(torch.autograd.Function):
    """Forward-only custom-autograd shell for one pressure time segment."""

    @staticmethod
    def forward(
        ctx,
        p: Tensor,
        u: Tensor,
        w: Tensor,
        src_v: Tensor,
        src_x: Tensor,
        src_z: Tensor,
        src_index: Tensor,
        rcv_x: Tensor,
        rcv_z: Tensor,
        kappa1: Tensor,
        alpha1: Tensor,
        kappa2: Tensor,
        alpha2: Tensor,
        kappa3: Tensor,
        nx: int,
        nz: int,
        dx: float,
        dz: float,
        dt: float,
        nabc: int,
        free_surface: bool,
    ):
        del ctx
        p_out, u_out, w_out, rcv_p, _ = step_forward_pressure_only(
            nx,
            nz,
            dx,
            dz,
            dt,
            nabc,
            free_surface,
            src_x,
            src_z,
            int(src_x.numel()),
            src_index,
            src_v,
            rcv_x,
            rcv_z,
            int(rcv_x.numel()),
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            9.0 / 8.0,
            -1.0 / 24.0,
            p,
            u,
            w,
            save_forward_wavefield=False,
            accumulate_wavefield_in_grad=False,
            device=p.device,
            dtype=p.dtype,
        )
        return p_out, u_out, w_out, rcv_p

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError(
            "custom_autograd_forward segment backend only implements forward "
            "parity; backward/gradient policy is not implemented yet"
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
