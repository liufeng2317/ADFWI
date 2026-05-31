"""Opt-in custom-autograd acoustic propagation kernels.

These kernels are not the default acoustic propagator path. They exist as a
guarded performance option for receiver-loss workflows after the benchmark-only
chunk prototype passed output/loss and gradient parity gates.
"""

from typing import Dict

import torch

from .acoustic_kernels import pad_torchSingle


def _step_forward_saved(
    p,
    u,
    w,
    kappa1,
    alpha1,
    kappa2,
    alpha2,
    kappa3,
    *,
    free_surface_start: int,
    source_x,
    source_z,
    source_value,
    use_free_surface: bool,
):
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]

    p_new = p.clone()
    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    div_p = (
        c1
        * (
            u[:, zp, 2 : nx_pml - 2]
            - u[:, zp, 1 : nx_pml - 3]
            + w[:, zp, 2 : nx_pml - 2]
            - w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2]
        )
        + c2
        * (
            u[:, zp, 3 : nx_pml - 1]
            - u[:, zp, 0 : nx_pml - 4]
            + w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2]
            - w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2]
        )
    )
    p_new[:, zp, xp] = (1.0 - kappa1[zp, xp]) * p[:, zp, xp] - alpha1[zp, xp] * div_p
    source_index = torch.arange(p_new.shape[0], device=p_new.device)
    p_new[source_index, source_z, source_x] = p_new[source_index, source_z, source_x] + source_value
    if use_free_surface:
        p_new[:, free_surface_start - 1, :] = -p_new[:, free_surface_start + 1, :]

    u_new = u.clone()
    zu = slice(free_surface_start, nz_pml - 1)
    xu = slice(1, nx_pml - 2)
    div_u = (
        c1 * (p_new[:, zu, 2 : nx_pml - 1] - p_new[:, zu, 1 : nx_pml - 2])
        + c2 * (p_new[:, zu, 3:nx_pml] - p_new[:, zu, 0 : nx_pml - 3])
    )
    u_new[:, zu, xu] = (1.0 - kappa2[zu, xu]) * u[:, zu, xu] - alpha2[zu, xu] * div_u

    w_new = w.clone()
    zw = slice(free_surface_start, nz_pml - 2)
    xw = slice(1, nx_pml - 1)
    div_w = (
        c1 * (p_new[:, free_surface_start + 1 : nz_pml - 1, xw] - p_new[:, zw, xw])
        + c2
        * (
            p_new[:, free_surface_start + 2 : nz_pml, xw]
            - p_new[:, free_surface_start - 1 : nz_pml - 3, xw]
        )
    )
    w_new[:, zw, xw] = (1.0 - kappa3[zw, xw]) * w[:, zw, xw] - alpha2[zw, xw] * div_w
    if use_free_surface:
        w_new[:, free_surface_start - 1, :] = w_new[:, free_surface_start, :]
    return p_new, u_new, w_new, div_p, div_u, div_w


def _step_backward_saved(
    p,
    u,
    w,
    kappa1,
    alpha1,
    kappa2,
    alpha2,
    kappa3,
    div_p,
    div_u,
    div_w,
    grad_p_out,
    grad_u_out,
    grad_w_out,
    *,
    free_surface_start: int,
    use_free_surface: bool,
):
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]

    grad_p_new = grad_p_out.clone()
    grad_u = grad_u_out.clone()
    grad_w = grad_w_out.clone()
    grad_kappa1 = torch.zeros_like(kappa1)
    grad_alpha1 = torch.zeros_like(alpha1)
    grad_kappa2 = torch.zeros_like(kappa2)
    grad_alpha2 = torch.zeros_like(alpha2)
    grad_kappa3 = torch.zeros_like(kappa3)

    if use_free_surface:
        grad_w[:, free_surface_start, :] += grad_w[:, free_surface_start - 1, :]
        grad_w[:, free_surface_start - 1, :] = 0.0

    zu = slice(free_surface_start, nz_pml - 1)
    xu = slice(1, nx_pml - 2)
    gu = grad_u_out[:, zu, xu]
    grad_u[:, zu, xu] = gu * (1.0 - kappa2[zu, xu])
    grad_kappa2[zu, xu] += torch.sum(-u[:, zu, xu] * gu, dim=0)
    grad_alpha2[zu, xu] += torch.sum(-div_u * gu, dim=0)
    grad_div_u = -alpha2[zu, xu].unsqueeze(0) * gu
    grad_p_new[:, zu, 2 : nx_pml - 1] += c1 * grad_div_u
    grad_p_new[:, zu, 1 : nx_pml - 2] -= c1 * grad_div_u
    grad_p_new[:, zu, 3:nx_pml] += c2 * grad_div_u
    grad_p_new[:, zu, 0 : nx_pml - 3] -= c2 * grad_div_u

    zw = slice(free_surface_start, nz_pml - 2)
    xw = slice(1, nx_pml - 1)
    gw = grad_w[:, zw, xw].clone()
    grad_w[:, zw, xw] = gw * (1.0 - kappa3[zw, xw])
    grad_kappa3[zw, xw] += torch.sum(-w[:, zw, xw] * gw, dim=0)
    grad_alpha2[zw, xw] += torch.sum(-div_w * gw, dim=0)
    grad_div_w = -alpha2[zw, xw].unsqueeze(0) * gw
    grad_p_new[:, free_surface_start + 1 : nz_pml - 1, xw] += c1 * grad_div_w
    grad_p_new[:, zw, xw] -= c1 * grad_div_w
    grad_p_new[:, free_surface_start + 2 : nz_pml, xw] += c2 * grad_div_w
    grad_p_new[:, free_surface_start - 1 : nz_pml - 3, xw] -= c2 * grad_div_w

    if use_free_surface:
        grad_p_new[:, free_surface_start + 1, :] -= grad_p_new[:, free_surface_start - 1, :]
        grad_p_new[:, free_surface_start - 1, :] = 0.0

    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    gp = grad_p_new[:, zp, xp]
    grad_p = grad_p_new.clone()
    grad_p[:, zp, xp] = gp * (1.0 - kappa1[zp, xp])
    grad_kappa1[zp, xp] = torch.sum(-p[:, zp, xp] * gp, dim=0)
    grad_alpha1[zp, xp] = torch.sum(-div_p * gp, dim=0)
    grad_div_p = -alpha1[zp, xp].unsqueeze(0) * gp
    grad_u[:, zp, 2 : nx_pml - 2] += c1 * grad_div_p
    grad_u[:, zp, 1 : nx_pml - 3] -= c1 * grad_div_p
    grad_u[:, zp, 3 : nx_pml - 1] += c2 * grad_div_p
    grad_u[:, zp, 0 : nx_pml - 4] -= c2 * grad_div_p
    grad_w[:, zp, 2 : nx_pml - 2] += c1 * grad_div_p
    grad_w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2] -= c1 * grad_div_p
    grad_w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2] += c2 * grad_div_p
    grad_w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2] -= c2 * grad_div_p

    return (
        grad_p,
        grad_u,
        grad_w,
        grad_kappa1,
        grad_alpha1,
        grad_kappa2,
        grad_alpha2,
        grad_kappa3,
    )


class _CustomChunkForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        p,
        u,
        w,
        kappa1,
        alpha1,
        kappa2,
        alpha2,
        kappa3,
        source_x,
        source_z,
        source_v,
        rcv_x,
        rcv_z,
        free_surface_start: int,
        use_free_surface: bool,
    ):
        p_states = []
        u_states = []
        w_states = []
        div_p_values = []
        div_u_values = []
        div_w_values = []
        records_p = []
        records_u = []
        records_w = []

        for step in range(source_v.shape[0]):
            p_states.append(p)
            u_states.append(u)
            w_states.append(w)
            p, u, w, div_p, div_u, div_w = _step_forward_saved(
                p,
                u,
                w,
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                free_surface_start=free_surface_start,
                source_x=source_x,
                source_z=source_z,
                source_value=source_v[step],
                use_free_surface=use_free_surface,
            )
            div_p_values.append(div_p)
            div_u_values.append(div_u)
            div_w_values.append(div_w)
            records_p.append(p[:, rcv_z, rcv_x])
            records_u.append(u[:, rcv_z, rcv_x])
            records_w.append(w[:, rcv_z, rcv_x])

        ctx.free_surface_start = free_surface_start
        ctx.use_free_surface = use_free_surface
        ctx.save_for_backward(
            torch.stack(p_states),
            torch.stack(u_states),
            torch.stack(w_states),
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            torch.stack(div_p_values),
            torch.stack(div_u_values),
            torch.stack(div_w_values),
            rcv_x,
            rcv_z,
        )
        return p, u, w, torch.stack(records_p, dim=1), torch.stack(records_u, dim=1), torch.stack(records_w, dim=1)

    @staticmethod
    def backward(ctx, grad_p, grad_u, grad_w, grad_rcv_p, grad_rcv_u, grad_rcv_w):
        (
            p_states,
            u_states,
            w_states,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            div_p_values,
            div_u_values,
            div_w_values,
            rcv_x,
            rcv_z,
        ) = ctx.saved_tensors
        grad_kappa1 = torch.zeros_like(kappa1)
        grad_alpha1 = torch.zeros_like(alpha1)
        grad_kappa2 = torch.zeros_like(kappa2)
        grad_alpha2 = torch.zeros_like(alpha2)
        grad_kappa3 = torch.zeros_like(kappa3)

        for step in range(p_states.shape[0] - 1, -1, -1):
            grad_p[:, rcv_z, rcv_x] += grad_rcv_p[:, step, :]
            grad_u[:, rcv_z, rcv_x] += grad_rcv_u[:, step, :]
            grad_w[:, rcv_z, rcv_x] += grad_rcv_w[:, step, :]
            (
                grad_p,
                grad_u,
                grad_w,
                step_grad_kappa1,
                step_grad_alpha1,
                step_grad_kappa2,
                step_grad_alpha2,
                step_grad_kappa3,
            ) = _step_backward_saved(
                p_states[step],
                u_states[step],
                w_states[step],
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                div_p_values[step],
                div_u_values[step],
                div_w_values[step],
                grad_p,
                grad_u,
                grad_w,
                free_surface_start=ctx.free_surface_start,
                use_free_surface=ctx.use_free_surface,
            )
            grad_kappa1 += step_grad_kappa1
            grad_alpha1 += step_grad_alpha1
            grad_kappa2 += step_grad_kappa2
            grad_alpha2 += step_grad_alpha2
            grad_kappa3 += step_grad_kappa3

        return (
            grad_p,
            grad_u,
            grad_w,
            grad_kappa1,
            grad_alpha1,
            grad_kappa2,
            grad_alpha2,
            grad_kappa3,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class _RematerializedCustomChunkForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        p,
        u,
        w,
        kappa1,
        alpha1,
        kappa2,
        alpha2,
        kappa3,
        source_x,
        source_z,
        source_v,
        rcv_x,
        rcv_z,
        free_surface_start: int,
        use_free_surface: bool,
    ):
        records_p = []
        records_u = []
        records_w = []
        p_start = p
        u_start = u
        w_start = w

        for step in range(source_v.shape[0]):
            p, u, w, _, _, _ = _step_forward_saved(
                p,
                u,
                w,
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                free_surface_start=free_surface_start,
                source_x=source_x,
                source_z=source_z,
                source_value=source_v[step],
                use_free_surface=use_free_surface,
            )
            records_p.append(p[:, rcv_z, rcv_x])
            records_u.append(u[:, rcv_z, rcv_x])
            records_w.append(w[:, rcv_z, rcv_x])

        ctx.free_surface_start = free_surface_start
        ctx.use_free_surface = use_free_surface
        ctx.save_for_backward(
            p_start,
            u_start,
            w_start,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            source_x,
            source_z,
            source_v,
            rcv_x,
            rcv_z,
        )
        return p, u, w, torch.stack(records_p, dim=1), torch.stack(records_u, dim=1), torch.stack(records_w, dim=1)

    @staticmethod
    def backward(ctx, grad_p, grad_u, grad_w, grad_rcv_p, grad_rcv_u, grad_rcv_w):
        (
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            source_x,
            source_z,
            source_v,
            rcv_x,
            rcv_z,
        ) = ctx.saved_tensors
        p_states = []
        u_states = []
        w_states = []
        div_p_values = []
        div_u_values = []
        div_w_values = []

        for step in range(source_v.shape[0]):
            p_states.append(p)
            u_states.append(u)
            w_states.append(w)
            p, u, w, div_p, div_u, div_w = _step_forward_saved(
                p,
                u,
                w,
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                free_surface_start=ctx.free_surface_start,
                source_x=source_x,
                source_z=source_z,
                source_value=source_v[step],
                use_free_surface=ctx.use_free_surface,
            )
            div_p_values.append(div_p)
            div_u_values.append(div_u)
            div_w_values.append(div_w)

        grad_kappa1 = torch.zeros_like(kappa1)
        grad_alpha1 = torch.zeros_like(alpha1)
        grad_kappa2 = torch.zeros_like(kappa2)
        grad_alpha2 = torch.zeros_like(alpha2)
        grad_kappa3 = torch.zeros_like(kappa3)

        for step in range(len(p_states) - 1, -1, -1):
            grad_p[:, rcv_z, rcv_x] += grad_rcv_p[:, step, :]
            grad_u[:, rcv_z, rcv_x] += grad_rcv_u[:, step, :]
            grad_w[:, rcv_z, rcv_x] += grad_rcv_w[:, step, :]
            (
                grad_p,
                grad_u,
                grad_w,
                step_grad_kappa1,
                step_grad_alpha1,
                step_grad_kappa2,
                step_grad_alpha2,
                step_grad_kappa3,
            ) = _step_backward_saved(
                p_states[step],
                u_states[step],
                w_states[step],
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                div_p_values[step],
                div_u_values[step],
                div_w_values[step],
                grad_p,
                grad_u,
                grad_w,
                free_surface_start=ctx.free_surface_start,
                use_free_surface=ctx.use_free_surface,
            )
            grad_kappa1 += step_grad_kappa1
            grad_alpha1 += step_grad_alpha1
            grad_kappa2 += step_grad_kappa2
            grad_alpha2 += step_grad_alpha2
            grad_kappa3 += step_grad_kappa3

        return (
            grad_p,
            grad_u,
            grad_w,
            grad_kappa1,
            grad_alpha1,
            grad_kappa2,
            grad_alpha2,
            grad_kappa3,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def custom_chunk_forward_kernel(
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
    checkpoint_segments: int = 1,
    save_forward_wavefield: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run the guarded custom-chunk acoustic path.

    This opt-in path is intentionally narrower than ``forward_kernel``.
    ``checkpoint_segments`` controls custom chunk segmentation here; it is not
    PyTorch checkpoint rematerialization.
    """
    if checkpoint_segments < 1:
        raise ValueError("checkpoint_segments must be positive")
    if save_forward_wavefield:
        raise ValueError("custom chunk acoustic forward does not support save_forward_wavefield=True")
    if src_v.shape != (src_n, nt):
        raise ValueError(f"expected src_v shape ({src_n}, {nt}), got {tuple(src_v.shape)}")
    if dx <= 0 or dz <= 0:
        raise ValueError("dx and dz must be positive")
    if rcv_n != rcv_x.numel() or rcv_n != rcv_z.numel():
        raise ValueError("rcv_n must match receiver coordinate lengths")

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

    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    step = 0
    for chunk in torch.chunk(src_v, checkpoint_segments, dim=-1):
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp = _CustomChunkForward.apply(
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            src_x + nabc,
            src_z + nabc,
            (dt * chunk).transpose(0, 1).contiguous(),
            rcv_x + nabc,
            rcv_z + nabc,
            free_surface_start,
            free_surface,
        )
        next_step = step + chunk.shape[-1]
        rcv_p[:, step:next_step] = rcv_p_temp
        rcv_u[:, step:next_step] = rcv_u_temp
        rcv_w[:, step:next_step] = rcv_w_temp
        step = next_step

    return {
        "p": rcv_p,
        "u": rcv_u,
        "w": rcv_w,
        "forward_wavefield_p": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_u": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_w": torch.zeros((nz, nx), dtype=dtype, device=device),
    }


def rematerialized_custom_chunk_forward_kernel(
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
    checkpoint_segments: int = 1,
    save_forward_wavefield: bool = False,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run a checkpoint-compatible custom acoustic chunk prototype.

    The forward pass saves only each chunk's boundary state and rematerializes
    chunk internals during backward. This is experimental and intentionally not
    wired into the default acoustic propagator path.
    """
    if checkpoint_segments < 1:
        raise ValueError("checkpoint_segments must be positive")
    if save_forward_wavefield:
        raise ValueError("rematerialized custom chunk acoustic forward does not support save_forward_wavefield=True")
    if src_v.shape != (src_n, nt):
        raise ValueError(f"expected src_v shape ({src_n}, {nt}), got {tuple(src_v.shape)}")
    if dx <= 0 or dz <= 0:
        raise ValueError("dx and dz must be positive")
    if rcv_n != rcv_x.numel() or rcv_n != rcv_z.numel():
        raise ValueError("rcv_n must match receiver coordinate lengths")

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

    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    step = 0
    for chunk in torch.chunk(src_v, checkpoint_segments, dim=-1):
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp = _RematerializedCustomChunkForward.apply(
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            src_x + nabc,
            src_z + nabc,
            (dt * chunk).transpose(0, 1).contiguous(),
            rcv_x + nabc,
            rcv_z + nabc,
            free_surface_start,
            free_surface,
        )
        next_step = step + chunk.shape[-1]
        rcv_p[:, step:next_step] = rcv_p_temp
        rcv_u[:, step:next_step] = rcv_u_temp
        rcv_w[:, step:next_step] = rcv_w_temp
        step = next_step

    return {
        "p": rcv_p,
        "u": rcv_u,
        "w": rcv_w,
        "forward_wavefield_p": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_u": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_w": torch.zeros((nz, nx), dtype=dtype, device=device),
    }
