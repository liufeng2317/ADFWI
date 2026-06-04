#!/usr/bin/env python
"""Benchmark a chunk-level custom-autograd acoustic recurrence prototype.

This script stays under ``scripts/benchmark`` and does not modify production
propagator code. It tests whether one custom autograd dispatch for a whole time
chunk can preserve the already validated timestep formulas while avoiding the
per-timestep ``autograd.Function.apply`` overhead of the earlier prototype.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from scripts.benchmark.acoustic_custom_multistep_update_probe import (
    build_inputs,
    receiver_indices,
    source_values,
    tensor_diff,
    timestep_reference_with_features,
)
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_custom_chunk_forward_20260531.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


class Timer:
    def __init__(self, backend) -> None:
        self.backend = backend

    def measure(self, fn):
        synchronize(self.backend)
        start = time.perf_counter()
        value = fn()
        synchronize(self.backend)
        return value, time.perf_counter() - start


def step_forward_saved(
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
    use_source: bool,
    use_free_surface: bool,
):
    """Run one acoustic step and return intermediate divergences for backward."""
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
    if use_source:
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


def step_backward_saved(
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
    """Manual adjoint for one saved acoustic step."""
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


class CustomChunkForward(torch.autograd.Function):
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
            p, u, w, div_p, div_u, div_w = step_forward_saved(
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
                use_source=True,
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
            ) = step_backward_saved(
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


def run_reference(inputs, args: argparse.Namespace, *, source_x, source_z, source_v, rcv_x, rcv_z):
    p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3 = inputs
    records_p = []
    records_u = []
    records_w = []
    for step in range(args.steps):
        p, u, w = timestep_reference_with_features(
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            free_surface_start=args.nabc,
            source_x=source_x,
            source_z=source_z,
            source_value=source_v[step],
            use_source=True,
            use_free_surface=args.free_surface_boundary_write,
        )
        records_p.append(p[:, rcv_z, rcv_x])
        records_u.append(u[:, rcv_z, rcv_x])
        records_w.append(w[:, rcv_z, rcv_x])
    return p, u, w, torch.stack(records_p, dim=1), torch.stack(records_u, dim=1), torch.stack(records_w, dim=1)


def run_custom(inputs, args: argparse.Namespace, *, source_x, source_z, source_v, rcv_x, rcv_z):
    return CustomChunkForward.apply(
        *inputs,
        source_x,
        source_z,
        source_v,
        rcv_x,
        rcv_z,
        args.nabc,
        args.free_surface_boundary_write,
    )


def build_loss(outputs, args: argparse.Namespace):
    if args.loss_kind == "energy":
        return sum(output.pow(2).mean() for output in outputs)
    if args.loss_kind == "receiver-random-linear":
        torch.manual_seed(args.upstream_seed)
        loss = None
        for name, output in zip(("p", "u", "w", "rcv_p", "rcv_u", "rcv_w"), outputs):
            if name not in args.loss_components:
                continue
            upstream = args.upstream_scale * torch.randn_like(output)
            term = (output * upstream).sum()
            loss = term if loss is None else loss + term
        if loss is None:
            raise ValueError("no active loss components")
        return loss
    raise ValueError(f"unknown loss kind: {args.loss_kind}")


def run_variant(args: argparse.Namespace, backend, *, custom: bool) -> Dict[str, Any]:
    timer = Timer(backend)
    inputs = build_inputs(args, device=backend.device, dtype=backend.dtype)
    rcv_z, rcv_x = receiver_indices(args, device=backend.device)
    source_x = torch.full((args.shots,), args.nabc + args.nx // 2, device=backend.device, dtype=torch.long)
    source_z = torch.full((args.shots,), args.nabc + args.source_depth, device=backend.device, dtype=torch.long)
    source_v = source_values(args, device=backend.device, dtype=backend.dtype)
    runner = run_custom if custom else run_reference
    outputs, forward_seconds = timer.measure(
        lambda: runner(inputs, args, source_x=source_x, source_z=source_z, source_v=source_v, rcv_x=rcv_x, rcv_z=rcv_z)
    )
    loss = build_loss(outputs, args)
    _, backward_seconds = timer.measure(lambda: loss.backward())
    grads = [
        tensor.grad.detach().clone() if tensor.grad is not None else torch.zeros_like(tensor)
        for tensor in inputs
    ]
    return {
        "outputs": [output.detach() for output in outputs],
        "loss": float(loss.detach().cpu().item()),
        "grads": grads,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
    }


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    output_names = ("p", "u", "w", "rcv_p", "rcv_u", "rcv_w")
    grad_names = ("p0", "u0", "w0", "kappa1", "alpha1", "kappa2", "alpha2", "kappa3")
    return {
        "loss_abs_diff": abs(candidate["loss"] - reference["loss"]),
        "outputs": {
            name: tensor_diff(ref, cand)
            for name, ref, cand in zip(output_names, reference["outputs"], candidate["outputs"])
        },
        "grads": {
            name: tensor_diff(ref, cand)
            for name, ref, cand in zip(grad_names, reference["grads"], candidate["grads"])
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    data = list(values)
    return {"min": float(min(data)), "max": float(max(data)), "mean": float(sum(data) / len(data)), "values": data}


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(requested_device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)
    pairs = []
    for index in range(args.warmup + args.repeat):
        reference = run_variant(args, backend, custom=False)
        candidate = run_variant(args, backend, custom=True)
        if index >= args.warmup:
            pairs.append(
                {
                    "reference": {key: value for key, value in reference.items() if key not in {"outputs", "grads"}},
                    "candidate": {key: value for key, value in candidate.items() if key not in {"outputs", "grads"}},
                    "comparison": compare_pair(reference, candidate),
                }
            )
    return {
        "status": "ok",
        "purpose": "chunk-level custom autograd feasibility probe for acoustic recurrence",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "steps": args.steps,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "pressure_scale": args.pressure_scale,
            "velocity_scale": args.velocity_scale,
            "kappa_scale": args.kappa_scale,
            "alpha1_scale": args.alpha1_scale,
            "alpha2_scale": args.alpha2_scale,
            "source_depth": args.source_depth,
            "source_scale": args.source_scale,
            "free_surface_boundary_write": args.free_surface_boundary_write,
            "receivers": args.receivers,
            "receiver_depth": args.receiver_depth,
            "loss_kind": args.loss_kind,
            "loss_components": args.loss_components,
            "upstream_seed": args.upstream_seed,
            "upstream_scale": args.upstream_scale,
        },
        "summary": {
            "speedup": {
                "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pairs),
                "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pairs),
                "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pairs),
            },
            "max_differences": {
                "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
                "output_max_abs_diff": max(
                    metric["max_abs_diff"] for pair in pairs for metric in pair["comparison"]["outputs"].values()
                ),
                "output_max_rel_diff": max(
                    metric["max_rel_diff"] for pair in pairs for metric in pair["comparison"]["outputs"].values()
                ),
                "grad_max_abs_diff": max(
                    metric["max_abs_diff"] for pair in pairs for metric in pair["comparison"]["grads"].values()
                ),
                "grad_max_rel_diff": max(
                    metric["max_rel_diff"] for pair in pairs for metric in pair["comparison"]["grads"].values()
                ),
            },
        },
        "pairs": pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240531)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--pressure-scale", type=float, default=1e-3)
    parser.add_argument("--velocity-scale", type=float, default=1e-6)
    parser.add_argument("--kappa-scale", type=float, default=1e-3)
    parser.add_argument("--alpha1-scale", type=float, default=1e-3)
    parser.add_argument("--alpha2-scale", type=float, default=1e-3)
    parser.add_argument("--source-depth", type=int, default=1)
    parser.add_argument("--source-scale", type=float, default=1e-4)
    parser.add_argument("--free-surface-boundary-write", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receivers", type=int, default=8)
    parser.add_argument("--receiver-depth", type=int, default=1)
    parser.add_argument("--loss-kind", choices=("energy", "receiver-random-linear"), default="energy")
    parser.add_argument(
        "--loss-components",
        default="p,u,w,rcv_p,rcv_u,rcv_w",
        help="Comma-separated outputs used by --loss-kind receiver-random-linear.",
    )
    parser.add_argument("--upstream-seed", type=int, default=20240601)
    parser.add_argument("--upstream-scale", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.shots <= 0:
        parser.error("--shots must be positive")
    if args.receivers <= 0:
        parser.error("--receivers must be positive")
    args.loss_components = tuple(item.strip() for item in args.loss_components.split(",") if item.strip())
    if not args.loss_components:
        parser.error("--loss-components must not be empty")
    unsupported = set(args.loss_components) - {"p", "u", "w", "rcv_p", "rcv_u", "rcv_w"}
    if unsupported:
        parser.error(f"unsupported --loss-components: {sorted(unsupported)}")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
