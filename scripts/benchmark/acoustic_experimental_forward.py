"""Experimental acoustic forward path for performance benchmarks.

This module is intentionally located under ``scripts/benchmark``. It is not a
public ADFWI propagator API and is not imported by user examples. It exists to
validate whether the custom-gradient acoustic recurrence can match production
``forward_kernel`` before any production-facing implementation is considered.
"""

from __future__ import annotations

from typing import Dict

import torch

from ADFWI.propagator.acoustic_kernels import pad_torchSingle
from scripts.benchmark.acoustic_custom_multistep_update_probe import timestep_custom_with_features


def experimental_forward_kernel(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    nt: int,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: torch.Tensor,
    src_z: torch.Tensor,
    src_n: int,
    src_v: torch.Tensor,
    rcv_x: torch.Tensor,
    rcv_z: torch.Tensor,
    rcv_n: int,
    damp: torch.Tensor,
    v: torch.Tensor,
    rho: torch.Tensor,
    *,
    save_forward_wavefield: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run the experimental custom-gradient acoustic recurrence.

    Limitations are explicit by design: the path currently supports the
    benchmark contract used by the parity harness, not the full public kernel
    surface.
    """
    if dx <= 0 or dz <= 0:
        raise ValueError("dx and dz must be positive")
    if src_n <= 0:
        raise ValueError("src_n must be positive")
    if src_x.numel() != src_n or src_z.numel() != src_n:
        raise ValueError("src_x and src_z must contain one location per source")
    if src_v.shape != (src_n, nt):
        raise ValueError(f"expected src_v shape ({src_n}, {nt}), got {tuple(src_v.shape)}")
    if save_forward_wavefield:
        raise ValueError("experimental_forward_kernel does not yet support forward wavefield summaries")

    c = pad_torchSingle(v, nabc, nz, nx, src_n, device=device)
    den = pad_torchSingle(rho, nabc, nz, nx, src_n, device=device)
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    p = torch.zeros((src_n, nz_pml, nx_pml), dtype=dtype, device=device)
    u = torch.zeros((src_n, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((src_n, nz_pml - 1, nx_pml), dtype=dtype, device=device)
    free_surface_start = nabc if free_surface else 1

    alpha1 = den * c * c * dt / dz
    kappa1 = damp * dt
    alpha2 = dt / (den * dz)
    kappa2 = torch.zeros_like(damp, device=device)
    kappa2[:, 1 : nx_pml - 2] = 0.5 * (damp[:, 1 : nx_pml - 2] + damp[:, 2 : nx_pml - 1]) * dt
    kappa3 = torch.zeros_like(damp, device=device)
    kappa3[free_surface_start : nz_pml - 2, :] = (
        0.5
        * (
            damp[free_surface_start : nz_pml - 2, :]
            + damp[free_surface_start + 1 : nz_pml - 1, :]
        )
        * dt
    )

    src_x_pml = src_x + nabc
    src_z_pml = src_z + nabc
    rcv_x_pml = rcv_x + nabc
    rcv_z_pml = rcv_z + nabc
    records_p = []
    records_u = []
    records_w = []

    for it in range(nt):
        p, u, w = timestep_custom_with_features(
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            free_surface_start=free_surface_start,
            source_x=src_x_pml,
            source_z=src_z_pml,
            source_value=dt * src_v[:, it],
            use_source=True,
            use_free_surface=free_surface,
        )
        records_p.append(p[:, rcv_z_pml, rcv_x_pml])
        records_u.append(u[:, rcv_z_pml, rcv_x_pml])
        records_w.append(w[:, rcv_z_pml, rcv_x_pml])

    return {
        "p": torch.stack(records_p, dim=1),
        "u": torch.stack(records_u, dim=1),
        "w": torch.stack(records_w, dim=1),
        "forward_wavefield_p": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_u": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_w": torch.zeros((nz, nx), dtype=dtype, device=device),
    }
