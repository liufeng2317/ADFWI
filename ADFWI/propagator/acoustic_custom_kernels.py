"""Opt-in custom-autograd acoustic propagation kernels.

Description
--------------
    This module contains experimental acoustic kernels with manually defined
    PyTorch autograd for chunk-level propagation. They are not the default
    acoustic propagator path in ``acoustic_kernels.py``.

    The default kernel relies on PyTorch autograd through the full time
    recurrence. The kernels here keep the same finite-difference update
    equations, but replace part of the backward graph with explicit adjoint
    updates inside ``torch.autograd.Function``.

Optimization boundary
--------------
    These paths are opt-in only. They are intended for measured receiver-loss
    workflows where forward wavefield summaries are not required. Do not use
    them as a drop-in replacement for the default kernel unless forward
    waveforms, loss, and raw model gradients have been compared for the target
    case.
"""

from typing import Dict, Tuple

import torch

from .acoustic_kernels import pad_torchSingle


@torch.jit.script
def _remat_pressure_forward_chunk_script(
    p: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    kappa1: torch.Tensor,
    alpha1: torch.Tensor,
    kappa2: torch.Tensor,
    alpha2: torch.Tensor,
    kappa3: torch.Tensor,
    source_x: torch.Tensor,
    source_z: torch.Tensor,
    source_v: torch.Tensor,
    rcv_x: torch.Tensor,
    rcv_z: torch.Tensor,
    free_surface_start: int,
    use_free_surface: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Scripted pressure-only remat forward loop.

    This helper mirrors the forward equations used by the Python remat step,
    but keeps the pressure-only receiver loop in TorchScript. The custom
    backward still owns gradient replay; this function only reduces forward
    Python/list/stack overhead for the memory-budget pressure-loss path.
    """
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nt = source_v.shape[0]
    src_n = p.shape[0]
    rcv_n = rcv_x.numel()
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]
    source_index = torch.arange(src_n, dtype=torch.long, device=p.device)
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=p.dtype, device=p.device)

    for step in range(nt):
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
        p_new[source_index, source_z, source_x] = p_new[source_index, source_z, source_x] + source_v[step]
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

        p = p_new
        u = u_new
        w = w_new
        rcv_p[:, step, :] = p[:, rcv_z, rcv_x]
    return p, u, w, rcv_p


def _parse_divergence_cache_components(components) -> frozenset[str]:
    """Parse the rematerialized backward divergence-cache selection.

    Parameters:
    --------------
        components : None, str, or iterable
            Component names to cache during backward replay. Valid component
            names are ``p``, ``u``, and ``w``.

    Returns:
    ------------------
        frozenset[str]
            Normalized component names.
    """
    if components is None:
        return frozenset({"p", "u", "w"})
    if isinstance(components, str):
        text = components.strip().lower()
        if text in {"", "none", "false", "0"}:
            return frozenset()
        parts = text.replace(",", " ").split()
    else:
        parts = [str(item).strip().lower() for item in components]
    allowed = {"p", "u", "w"}
    selected = frozenset(parts)
    invalid = selected - allowed
    if invalid:
        raise ValueError(f"unknown divergence cache component(s): {sorted(invalid)}")
    return selected


def _forward_step_with_saved_divergence(
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
    """Run one acoustic time step and return divergence intermediates.

    Description
    --------------
        This is the custom-autograd counterpart of one iteration in
        ``acoustic_kernels.step_forward``. It updates pressure ``p`` first,
        injects the source, applies the optional free-surface condition, then
        updates particle velocities ``u`` and ``w``.

        The returned divergence terms are saved because the manual backward
        formulas need them for gradients with respect to ``alpha`` and
        ``kappa`` coefficients.

    Returns:
    ------------------
        p_new, u_new, w_new
            Updated wavefield states.
        div_p, div_u, div_w
            Finite-difference divergence terms used by the adjoint update.
    """
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


def _pressure_divergence_from_state(
    p,
    u,
    w,
    *,
    free_surface_start: int,
):
    """Recompute the pressure-update divergence from a saved state."""
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]

    zp = slice(free_surface_start + 1, nz_pml - 2)
    return (
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


def _rebuild_pressure_from_divergence(
    p,
    kappa1,
    alpha1,
    div_p,
    *,
    free_surface_start: int,
    source_x,
    source_z,
    source_value,
    use_free_surface: bool,
):
    """Rebuild ``p_new`` from a saved state and a pressure divergence term."""
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]
    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    p_new = p.clone()
    source_index = torch.arange(p_new.shape[0], device=p_new.device)
    p_new[:, zp, xp] = (1.0 - kappa1[zp, xp]) * p[:, zp, xp] - alpha1[zp, xp] * div_p
    p_new[source_index, source_z, source_x] = p_new[source_index, source_z, source_x] + source_value
    if use_free_surface:
        p_new[:, free_surface_start - 1, :] = -p_new[:, free_surface_start + 1, :]
    return p_new


def _horizontal_velocity_divergence(p_new, *, free_surface_start: int):
    """Recompute the horizontal-velocity divergence from updated pressure."""
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p_new.shape[1]
    nx_pml = p_new.shape[2]
    zu = slice(free_surface_start, nz_pml - 1)
    return (
        c1 * (p_new[:, zu, 2 : nx_pml - 1] - p_new[:, zu, 1 : nx_pml - 2])
        + c2 * (p_new[:, zu, 3:nx_pml] - p_new[:, zu, 0 : nx_pml - 3])
    )


def _vertical_velocity_divergence(p_new, *, free_surface_start: int):
    """Recompute the vertical-velocity divergence from updated pressure."""
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p_new.shape[1]
    nx_pml = p_new.shape[2]
    xw = slice(1, nx_pml - 1)
    zw = slice(free_surface_start, nz_pml - 2)
    return (
        c1 * (p_new[:, free_surface_start + 1 : nz_pml - 1, xw] - p_new[:, zw, xw])
        + c2
        * (
            p_new[:, free_surface_start + 2 : nz_pml, xw]
            - p_new[:, free_surface_start - 1 : nz_pml - 3, xw]
        )
    )


def _backward_step_from_saved_divergence(
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
    """Apply the manual adjoint update for one acoustic time step.

    Description
    --------------
        This function is the backward counterpart of
        ``_forward_step_with_saved_divergence``.
        It propagates gradients from ``p_new``, ``u_new``, and ``w_new`` back to
        the previous states and model coefficients.

        The order mirrors the forward dependency graph:

        1. receiver/free-surface gradient contributions are already included in
           the incoming gradients;
        2. velocity updates contribute to ``grad_p_new`` and coefficient
           gradients;
        3. pressure free-surface handling is reversed;
        4. pressure update contributes to previous ``p``, ``u``, ``w`` and
           coefficient gradients.

    Returns:
    ------------------
        Gradients for ``p``, ``u``, ``w``, ``kappa1``, ``alpha1``, ``kappa2``,
        ``alpha2``, and ``kappa3``.
    """
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


class _SavedStateChunkFunction(torch.autograd.Function):
    """Custom autograd for a fully saved acoustic chunk.

    Description
    --------------
        The forward pass stores every internal time-step state and divergence
        term for the chunk. The backward pass then walks the chunk in reverse
        and applies ``_backward_step_from_saved_divergence``.

        This path is faster for some receiver-loss workflows, but it stores
        more internal state than checkpointed production execution. It is used
        only behind explicit opt-in controls.
    """

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
        divergence_save_components,
    ):
        divergence_save_components = _parse_divergence_cache_components(divergence_save_components)
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
            p, u, w, div_p, div_u, div_w = _forward_step_with_saved_divergence(
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
            if "p" in divergence_save_components:
                div_p_values.append(div_p)
            if "u" in divergence_save_components:
                div_u_values.append(div_u)
            if "w" in divergence_save_components:
                div_w_values.append(div_w)
            records_p.append(p[:, rcv_z, rcv_x])
            records_u.append(u[:, rcv_z, rcv_x])
            records_w.append(w[:, rcv_z, rcv_x])

        if "p" in divergence_save_components:
            div_p_cache = torch.stack(div_p_values)
        else:
            div_p_cache = p.new_empty((0,))
        if "u" in divergence_save_components:
            div_u_cache = torch.stack(div_u_values)
        else:
            div_u_cache = u.new_empty((0,))
        if "w" in divergence_save_components:
            div_w_cache = torch.stack(div_w_values)
        else:
            div_w_cache = w.new_empty((0,))

        ctx.free_surface_start = free_surface_start
        ctx.use_free_surface = use_free_surface
        ctx.divergence_save_components = divergence_save_components
        ctx.save_for_backward(
            torch.stack(p_states),
            torch.stack(u_states),
            torch.stack(w_states),
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            div_p_cache,
            div_u_cache,
            div_w_cache,
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
            source_x,
            source_z,
            source_v,
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
            if "p" in ctx.divergence_save_components:
                div_p = div_p_values[step]
            else:
                div_p = _pressure_divergence_from_state(
                    p_states[step],
                    u_states[step],
                    w_states[step],
                    free_surface_start=ctx.free_surface_start,
                )
            if "u" in ctx.divergence_save_components:
                div_u = div_u_values[step]
            else:
                div_u = None
            if "w" in ctx.divergence_save_components:
                div_w = div_w_values[step]
            else:
                div_w = None
            if div_u is None or div_w is None:
                p_new = _rebuild_pressure_from_divergence(
                    p_states[step],
                    kappa1,
                    alpha1,
                    div_p,
                    free_surface_start=ctx.free_surface_start,
                    source_x=source_x,
                    source_z=source_z,
                    source_value=source_v[step],
                    use_free_surface=ctx.use_free_surface,
                )
                if div_u is None:
                    div_u = _horizontal_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
                if div_w is None:
                    div_w = _vertical_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
            (
                grad_p,
                grad_u,
                grad_w,
                step_grad_kappa1,
                step_grad_alpha1,
                step_grad_kappa2,
                step_grad_alpha2,
                step_grad_kappa3,
            ) = _backward_step_from_saved_divergence(
                p_states[step],
                u_states[step],
                w_states[step],
                kappa1,
                alpha1,
                kappa2,
                alpha2,
                kappa3,
                div_p,
                div_u,
                div_w,
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
            None,
        )


class _RematerializedChunkFunction(torch.autograd.Function):
    """Custom autograd for a rematerialized acoustic chunk.

    Description
    --------------
        The forward pass records receiver data but saves only chunk boundary
        state. The backward pass rebuilds internal states as needed, optionally
        caching selected divergence terms or boundary states.

        This is a memory/speed research path. It is intentionally separate from
        the default propagator because changing saved state changes the
        checkpoint and replay contract.
    """

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
        divergence_cache_stride: int,
        divergence_cache_components,
        state_cache_stride: int,
        record_velocity_receivers: bool,
    ):
        records_p = []
        records_u = []
        records_w = []
        p_start = p
        u_start = u
        w_start = w
        boundary_p_states = []
        boundary_u_states = []
        boundary_w_states = []

        use_scripted_pressure_forward = (not record_velocity_receivers) and state_cache_stride <= 1
        if use_scripted_pressure_forward:
            p, u, w, rcv_p = _remat_pressure_forward_chunk_script(
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
                free_surface_start,
                use_free_surface,
            )
        else:
            for step in range(source_v.shape[0]):
                if state_cache_stride > 1 and step % state_cache_stride == 0:
                    boundary_p_states.append(p)
                    boundary_u_states.append(u)
                    boundary_w_states.append(w)
                p, u, w, _, _, _ = _forward_step_with_saved_divergence(
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
                if record_velocity_receivers:
                    records_u.append(u[:, rcv_z, rcv_x])
                    records_w.append(w[:, rcv_z, rcv_x])

        if state_cache_stride > 1:
            boundary_p_cache = torch.stack(boundary_p_states)
            boundary_u_cache = torch.stack(boundary_u_states)
            boundary_w_cache = torch.stack(boundary_w_states)
        else:
            boundary_p_cache = p.new_empty((0,))
            boundary_u_cache = u.new_empty((0,))
            boundary_w_cache = w.new_empty((0,))

        ctx.free_surface_start = free_surface_start
        ctx.use_free_surface = use_free_surface
        ctx.divergence_cache_stride = divergence_cache_stride
        ctx.divergence_cache_components = _parse_divergence_cache_components(divergence_cache_components)
        ctx.state_cache_stride = state_cache_stride
        ctx.record_velocity_receivers = record_velocity_receivers
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
            boundary_p_cache,
            boundary_u_cache,
            boundary_w_cache,
        )
        if not use_scripted_pressure_forward:
            rcv_p = torch.stack(records_p, dim=1)
        if record_velocity_receivers:
            rcv_u = torch.stack(records_u, dim=1)
            rcv_w = torch.stack(records_w, dim=1)
        else:
            rcv_u = p.new_empty((p.shape[0], 0, 0))
            rcv_w = p.new_empty((p.shape[0], 0, 0))
        return p, u, w, rcv_p, rcv_u, rcv_w

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
            boundary_p_cache,
            boundary_u_cache,
            boundary_w_cache,
        ) = ctx.saved_tensors
        if ctx.state_cache_stride <= 1:
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
                p, u, w, div_p, div_u, div_w = _forward_step_with_saved_divergence(
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
                should_cache = ctx.divergence_cache_stride > 0 and step % ctx.divergence_cache_stride == 0
                if should_cache and "p" in ctx.divergence_cache_components:
                    div_p_values.append(div_p)
                else:
                    div_p_values.append(None)
                if should_cache and "u" in ctx.divergence_cache_components:
                    div_u_values.append(div_u)
                else:
                    div_u_values.append(None)
                if should_cache and "w" in ctx.divergence_cache_components:
                    div_w_values.append(div_w)
                else:
                    div_w_values.append(None)
        grad_kappa1 = torch.zeros_like(kappa1)
        grad_alpha1 = torch.zeros_like(alpha1)
        grad_kappa2 = torch.zeros_like(kappa2)
        grad_alpha2 = torch.zeros_like(alpha2)
        grad_kappa3 = torch.zeros_like(kappa3)

        if ctx.state_cache_stride <= 1:
            for step in range(len(p_states) - 1, -1, -1):
                grad_p[:, rcv_z, rcv_x] += grad_rcv_p[:, step, :]
                if ctx.record_velocity_receivers:
                    grad_u[:, rcv_z, rcv_x] += grad_rcv_u[:, step, :]
                    grad_w[:, rcv_z, rcv_x] += grad_rcv_w[:, step, :]
                div_p = div_p_values[step]
                div_u = div_u_values[step]
                div_w = div_w_values[step]
                if div_p is None:
                    div_p = _pressure_divergence_from_state(
                        p_states[step],
                        u_states[step],
                        w_states[step],
                        free_surface_start=ctx.free_surface_start,
                    )
                if div_u is None or div_w is None:
                    p_new = _rebuild_pressure_from_divergence(
                        p_states[step],
                        kappa1,
                        alpha1,
                        div_p,
                        free_surface_start=ctx.free_surface_start,
                        source_x=source_x,
                        source_z=source_z,
                        source_value=source_v[step],
                        use_free_surface=ctx.use_free_surface,
                    )
                    if div_u is None:
                        div_u = _horizontal_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
                    if div_w is None:
                        div_w = _vertical_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
                (
                    grad_p,
                    grad_u,
                    grad_w,
                    step_grad_kappa1,
                    step_grad_alpha1,
                    step_grad_kappa2,
                    step_grad_alpha2,
                    step_grad_kappa3,
                ) = _backward_step_from_saved_divergence(
                    p_states[step],
                    u_states[step],
                    w_states[step],
                    kappa1,
                    alpha1,
                    kappa2,
                    alpha2,
                    kappa3,
                    div_p,
                    div_u,
                    div_w,
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
        else:
            for block_index in range(boundary_p_cache.shape[0] - 1, -1, -1):
                block_start = block_index * ctx.state_cache_stride
                block_end = min(block_start + ctx.state_cache_stride, source_v.shape[0])
                p_local = boundary_p_cache[block_index]
                u_local = boundary_u_cache[block_index]
                w_local = boundary_w_cache[block_index]
                p_states = []
                u_states = []
                w_states = []
                div_p_values = []
                div_u_values = []
                div_w_values = []

                for step in range(block_start, block_end):
                    p_states.append(p_local)
                    u_states.append(u_local)
                    w_states.append(w_local)
                    p_local, u_local, w_local, div_p, div_u, div_w = _forward_step_with_saved_divergence(
                        p_local,
                        u_local,
                        w_local,
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
                    should_cache = ctx.divergence_cache_stride > 0 and step % ctx.divergence_cache_stride == 0
                    div_p_values.append(div_p if should_cache and "p" in ctx.divergence_cache_components else None)
                    div_u_values.append(div_u if should_cache and "u" in ctx.divergence_cache_components else None)
                    div_w_values.append(div_w if should_cache and "w" in ctx.divergence_cache_components else None)

                for local_index in range(len(p_states) - 1, -1, -1):
                    step = block_start + local_index
                    grad_p[:, rcv_z, rcv_x] += grad_rcv_p[:, step, :]
                    if ctx.record_velocity_receivers:
                        grad_u[:, rcv_z, rcv_x] += grad_rcv_u[:, step, :]
                        grad_w[:, rcv_z, rcv_x] += grad_rcv_w[:, step, :]
                    div_p = div_p_values[local_index]
                    div_u = div_u_values[local_index]
                    div_w = div_w_values[local_index]
                    if div_p is None:
                        div_p = _pressure_divergence_from_state(
                            p_states[local_index],
                            u_states[local_index],
                            w_states[local_index],
                            free_surface_start=ctx.free_surface_start,
                        )
                    if div_u is None or div_w is None:
                        p_new = _rebuild_pressure_from_divergence(
                            p_states[local_index],
                            kappa1,
                            alpha1,
                            div_p,
                            free_surface_start=ctx.free_surface_start,
                            source_x=source_x,
                            source_z=source_z,
                            source_value=source_v[step],
                            use_free_surface=ctx.use_free_surface,
                        )
                        if div_u is None:
                            div_u = _horizontal_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
                        if div_w is None:
                            div_w = _vertical_velocity_divergence(p_new, free_surface_start=ctx.free_surface_start)
                    (
                        grad_p,
                        grad_u,
                        grad_w,
                        step_grad_kappa1,
                        step_grad_alpha1,
                        step_grad_kappa2,
                        step_grad_alpha2,
                        step_grad_kappa3,
                    ) = _backward_step_from_saved_divergence(
                        p_states[local_index],
                        u_states[local_index],
                        w_states[local_index],
                        kappa1,
                        alpha1,
                        kappa2,
                        alpha2,
                        kappa3,
                        div_p,
                        div_u,
                        div_w,
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
    divergence_save_components="p,u,w",
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run the guarded custom-chunk acoustic path.

    Description
    --------------
        This function follows the public return contract of the acoustic
        propagator, but uses ``_SavedStateChunkFunction`` for each chunk.

        ``checkpoint_segments`` controls custom chunk segmentation here; it is
        not PyTorch checkpoint rematerialization.

    Parameters:
    --------------
        nx, nz, dx, dz, nt, dt, nabc, free_surface
            Acoustic grid and time-stepping settings.
        src_x, src_z, src_n, src_v
            Source coordinates and wavelet values.
        rcv_x, rcv_z, rcv_n
            Receiver coordinates.
        damp, v, rho
            Absorbing boundary and acoustic model parameters.
        checkpoint_segments
            Number of custom-autograd chunks.
        save_forward_wavefield
            Must be ``False``. This path does not produce illumination
            wavefield summaries.
        divergence_save_components
            Experimental state-memory control. Component subset of divergence
            terms to save: ``p``, ``u``, and/or ``w``. The default
            ``p,u,w`` preserves the current high-speed path. ``none`` saves
            only wavefield states and recomputes divergence terms in backward.

    Returns:
    ------------------
        dict
            Recorded receiver components ``p``, ``u``, ``w`` and zero-valued
            forward-wavefield summary placeholders.
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
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp = _SavedStateChunkFunction.apply(
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
            divergence_save_components,
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


def compressed_custom_chunk_forward_kernel(*args, **kwargs) -> Dict[str, torch.Tensor]:
    """Run the saved-state custom chunk path without saved divergence tensors.

    Description
    --------------
        This benchmark-only variant keeps every per-step wavefield state, so it
        avoids rematerialized full-chunk replay, but drops the saved
        ``div_p/div_u/div_w`` tensors and recomputes them during backward.

        It is used to test whether the high-memory saved-state custom path can
        be compressed without losing most of its backward speed advantage.
    """
    kwargs["divergence_save_components"] = "none"
    return custom_chunk_forward_kernel(*args, **kwargs)


def pressure_divergence_custom_chunk_forward_kernel(*args, **kwargs) -> Dict[str, torch.Tensor]:
    """Run the saved-state custom chunk path while saving only pressure divergence.

    Description
    --------------
        This benchmark-only middle point keeps all per-step wavefield states and
        ``div_p``. It recomputes ``div_u`` and ``div_w`` during backward. The
        goal is to recover more speed than the fully compressed path without
        returning to the full saved-divergence memory cost.
    """
    kwargs["divergence_save_components"] = "p"
    return custom_chunk_forward_kernel(*args, **kwargs)


def velocity_divergence_custom_chunk_forward_kernel(*args, **kwargs) -> Dict[str, torch.Tensor]:
    """Run the saved-state custom chunk path while saving velocity divergences.

    Description
    --------------
        This benchmark-only middle point keeps all per-step wavefield states and
        saves ``div_u/div_w``. It recomputes only ``div_p`` during backward.
        This avoids rebuilding ``p_new`` and recomputing velocity divergences,
        so it tests a higher-speed state-compression point than saving only
        ``div_p``.
    """
    kwargs["divergence_save_components"] = "u,w"
    return custom_chunk_forward_kernel(*args, **kwargs)


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
    divergence_cache_stride: int = 0,
    divergence_cache_components=None,
    state_cache_stride: int = 1,
    record_velocity_receivers: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Run a checkpoint-compatible custom acoustic chunk prototype.

    Description
    --------------
        The forward pass saves only each chunk's boundary state and
        rematerializes chunk internals during backward. Optional divergence and
        state caching tune the memory/recompute tradeoff.

        This is experimental and intentionally not wired into the default
        acoustic propagator path.

    Parameters:
    --------------
        divergence_cache_stride
            Cache divergence terms every N steps. ``0`` disables divergence
            caching.
        divergence_cache_components
            Component subset to cache: ``p``, ``u``, ``w``.
        state_cache_stride
            Save internal state every N steps during backward replay.
        record_velocity_receivers
            When ``False``, record only pressure receivers and return zero
            placeholders for receiver ``u/w``. This is for pressure-loss
            benchmarks only; propagation state ``u/w`` is still maintained.

    Returns:
    ------------------
        dict
            Recorded receiver components ``p``, ``u``, ``w`` and zero-valued
            forward-wavefield summary placeholders.
    """
    if checkpoint_segments < 1:
        raise ValueError("checkpoint_segments must be positive")
    if save_forward_wavefield:
        raise ValueError("rematerialized custom chunk acoustic forward does not support save_forward_wavefield=True")
    if divergence_cache_stride < 0:
        raise ValueError("divergence_cache_stride must be non-negative")
    if state_cache_stride < 1:
        raise ValueError("state_cache_stride must be positive")
    _parse_divergence_cache_components(divergence_cache_components)
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
    if record_velocity_receivers:
        rcv_u = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
        rcv_w = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    else:
        rcv_u = None
        rcv_w = None
    step = 0
    for chunk in torch.chunk(src_v, checkpoint_segments, dim=-1):
        p, u, w, rcv_p_temp, rcv_u_temp, rcv_w_temp = _RematerializedChunkFunction.apply(
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
            divergence_cache_stride,
            divergence_cache_components,
            state_cache_stride,
            record_velocity_receivers,
        )
        next_step = step + chunk.shape[-1]
        rcv_p[:, step:next_step] = rcv_p_temp
        if record_velocity_receivers:
            rcv_u[:, step:next_step] = rcv_u_temp
            rcv_w[:, step:next_step] = rcv_w_temp
        step = next_step
    if not record_velocity_receivers:
        # Preserve the legacy waveform dictionary shape without spending chunk
        # time recording velocity receiver traces that pressure-loss FWI ignores.
        rcv_u = torch.zeros_like(rcv_p)
        rcv_w = torch.zeros_like(rcv_p)

    return {
        "p": rcv_p,
        "u": rcv_u,
        "w": rcv_w,
        "forward_wavefield_p": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_u": torch.zeros((nz, nx), dtype=dtype, device=device),
        "forward_wavefield_w": torch.zeros((nz, nx), dtype=dtype, device=device),
    }


def rematerialized_pressure_custom_chunk_forward_kernel(*args, **kwargs) -> Dict[str, torch.Tensor]:
    """Run the rematerialized custom path with pressure receiver records only."""
    kwargs["record_velocity_receivers"] = False
    return rematerialized_custom_chunk_forward_kernel(*args, **kwargs)
