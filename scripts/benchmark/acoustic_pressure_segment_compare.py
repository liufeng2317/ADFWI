#!/usr/bin/env python
"""Compare production pressure-only segment with custom pressure-update segment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADFWI.backends import configure_backend
from ADFWI.propagator.acoustic_kernels import pad_torchSingle, step_forward_pressure_only
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_pressure_segment_compare_20260605.json"
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


class CustomPressureUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, u, w, kappa1, alpha1, free_surface_start: int):
        c1 = 9.0 / 8.0
        c2 = -1.0 / 24.0
        nz_pml = p.shape[1]
        nx_pml = p.shape[2]
        z = slice(free_surface_start + 1, nz_pml - 2)
        x = slice(2, nx_pml - 2)
        div = (
            c1
            * (
                u[:, z, 2 : nx_pml - 2]
                - u[:, z, 1 : nx_pml - 3]
                + w[:, z, 2 : nx_pml - 2]
                - w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2]
            )
            + c2
            * (
                u[:, z, 3 : nx_pml - 1]
                - u[:, z, 0 : nx_pml - 4]
                + w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2]
                - w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2]
            )
        )
        p_next = p.clone()
        p_next[:, z, x] = (1.0 - kappa1[z, x]) * p[:, z, x] - alpha1[z, x] * div
        ctx.free_surface_start = free_surface_start
        ctx.save_for_backward(p.clone(), u.clone(), w.clone(), kappa1, alpha1, div)
        return p_next

    @staticmethod
    def backward(ctx, grad_output):
        p, u, w, kappa1, alpha1, div = ctx.saved_tensors
        del u, w
        free_surface_start = ctx.free_surface_start
        c1 = 9.0 / 8.0
        c2 = -1.0 / 24.0
        nz_pml = p.shape[1]
        nx_pml = p.shape[2]
        z = slice(free_surface_start + 1, nz_pml - 2)
        x = slice(2, nx_pml - 2)

        interior_grad = grad_output[:, z, x]
        grad_p = grad_output.clone()
        grad_p[:, z, x] = interior_grad * (1.0 - kappa1[z, x])
        grad_div = -alpha1[z, x].unsqueeze(0) * interior_grad

        grad_u = torch.zeros(
            (p.shape[0], p.shape[1], p.shape[2] - 1),
            dtype=p.dtype,
            device=p.device,
        )
        grad_u[:, z, 2 : nx_pml - 2] += c1 * grad_div
        grad_u[:, z, 1 : nx_pml - 3] -= c1 * grad_div
        grad_u[:, z, 3 : nx_pml - 1] += c2 * grad_div
        grad_u[:, z, 0 : nx_pml - 4] -= c2 * grad_div

        grad_w = torch.zeros(
            (p.shape[0], p.shape[1] - 1, p.shape[2]),
            dtype=p.dtype,
            device=p.device,
        )
        grad_w[:, z, 2 : nx_pml - 2] += c1 * grad_div
        grad_w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2] -= c1 * grad_div
        grad_w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2] += c2 * grad_div
        grad_w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2] -= c2 * grad_div

        grad_kappa1 = torch.zeros_like(kappa1)
        grad_kappa1[z, x] = torch.sum(-p[:, z, x] * interior_grad, dim=0)

        grad_alpha1 = torch.zeros_like(alpha1)
        grad_alpha1[z, x] = torch.sum(-div * interior_grad, dim=0)
        return grad_p, grad_u, grad_w, grad_kappa1, grad_alpha1, None


def custom_pressure_update(p, u, w, kappa1, alpha1, *, free_surface_start: int):
    return CustomPressureUpdate.apply(p, u, w, kappa1, alpha1, free_surface_start)


def step_forward_pressure_only_custom_update(
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    dt: float,
    nabc: int,
    free_surface: bool,
    src_x: torch.Tensor,
    src_z: torch.Tensor,
    src_n: int,
    src_index: torch.Tensor,
    src_v: torch.Tensor,
    rcv_x: torch.Tensor,
    rcv_z: torch.Tensor,
    rcv_n: int,
    kappa1: torch.Tensor,
    alpha1: torch.Tensor,
    kappa2: torch.Tensor,
    alpha2: torch.Tensor,
    kappa3: torch.Tensor,
    c1_staggered: float,
    c2_staggered: float,
    p: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    save_forward_wavefield: bool = True,
    accumulate_wavefield_in_grad: bool = True,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    del dx, dz, c1_staggered, c2_staggered
    p = p.clone()
    u = u.clone()
    w = w.clone()

    nt = src_v.shape[-1]
    free_surface_start = nabc if free_surface else 1
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    rcv_p = torch.zeros((src_n, nt, rcv_n), dtype=dtype, device=device)
    forward_wavefield_p = torch.zeros((nz, nx), dtype=dtype, device=device)

    for it in range(nt):
        p = custom_pressure_update(p, u, w, kappa1, alpha1, free_surface_start=free_surface_start)
        if src_z.dim() == 1:
            src_update = dt * (src_v[it] if len(src_v.shape) == 1 else src_v[:, it])
            p[src_index, src_z, src_x] = p[src_index, src_z, src_x] + src_update
        else:
            for i in range(src_n):
                src_update = dt * (src_v[i, it] if len(src_v.shape) == 2 else src_v[i, :, it])
                p[i, src_z[i], src_x[i]] = p[i, src_z[i], src_x[i]] + src_update

        if free_surface:
            p[:, free_surface_start - 1, :] = -p[:, free_surface_start + 1, :]

        u[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2] = (
            (1.0 - kappa2[free_surface_start:nz_pml - 1, 1:nx_pml - 2])
            * u[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2]
            - alpha2[free_surface_start:nz_pml - 1, 1:nx_pml - 2]
            * (
                9.0
                / 8.0
                * (
                    p[:, free_surface_start:nz_pml - 1, 2:nx_pml - 1]
                    - p[:, free_surface_start:nz_pml - 1, 1:nx_pml - 2]
                )
                - 1.0
                / 24.0
                * (
                    p[:, free_surface_start:nz_pml - 1, 3:nx_pml]
                    - p[:, free_surface_start:nz_pml - 1, 0:nx_pml - 3]
                )
            )
        )
        w[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1] = (
            (1.0 - kappa3[free_surface_start:nz_pml - 2, 1:nx_pml - 1])
            * w[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1]
            - alpha2[free_surface_start:nz_pml - 2, 1:nx_pml - 1]
            * (
                9.0
                / 8.0
                * (
                    p[:, free_surface_start + 1:nz_pml - 1, 1:nx_pml - 1]
                    - p[:, free_surface_start:nz_pml - 2, 1:nx_pml - 1]
                )
                - 1.0
                / 24.0
                * (
                    p[:, free_surface_start + 2:nz_pml, 1:nx_pml - 1]
                    - p[:, free_surface_start - 1:nz_pml - 3, 1:nx_pml - 1]
                )
            )
        )
        if free_surface:
            w[:, free_surface_start - 1, :] = w[:, free_surface_start, :]

        rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
        if save_forward_wavefield and (accumulate_wavefield_in_grad or not torch.is_grad_enabled()):
            forward_wavefield_p = forward_wavefield_p + torch.sum(p * p, dim=0)[
                nabc:nabc + nz,
                nabc:nabc + nx,
            ].detach()

    return p, u, w, rcv_p, forward_wavefield_p


def make_case(args: argparse.Namespace, backend):
    torch.manual_seed(args.seed)
    device = backend.device
    dtype = backend.dtype
    nx_pml = args.nx + 2 * args.nabc
    nz_pml = args.nz + 2 * args.nabc

    vp_template = torch.full((args.nz, args.nx), args.vp, dtype=dtype, device=device)
    rho_template = torch.full((args.nz, args.nx), args.rho, dtype=dtype, device=device)
    damp = torch.zeros((nz_pml, nx_pml), dtype=dtype, device=device)

    src_x = torch.linspace(2, args.nx - 3, args.shots, device=device).round().long() + args.nabc
    src_z = torch.full((args.shots,), 2 + args.nabc, dtype=torch.long, device=device)
    rcv_x = torch.linspace(1, args.nx - 2, args.receivers, device=device).round().long() + args.nabc
    rcv_z = torch.full((args.receivers,), 2 + args.nabc, dtype=torch.long, device=device)
    src_index = torch.arange(args.shots, dtype=torch.long, device=device)

    src_v = torch.zeros((args.shots, args.nt), dtype=dtype, device=device)
    src_v[:, min(max(args.source_time, 0), args.nt - 1)] = args.source_amplitude

    p = torch.zeros((args.shots, nz_pml, nx_pml), dtype=dtype, device=device)
    u = torch.zeros((args.shots, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((args.shots, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    return {
        "vp_template": vp_template,
        "rho_template": rho_template,
        "damp": damp,
        "src_x": src_x,
        "src_z": src_z,
        "src_index": src_index,
        "src_v": src_v,
        "rcv_x": rcv_x,
        "rcv_z": rcv_z,
        "p": p,
        "u": u,
        "w": w,
    }


def build_coefficients(args: argparse.Namespace, backend, case, vp, rho):
    c = pad_torchSingle(vp, args.nabc, args.nz, args.nx, args.shots, device=backend.device)
    den = pad_torchSingle(rho, args.nabc, args.nz, args.nx, args.shots, device=backend.device)
    nx_pml = args.nx + 2 * args.nabc
    nz_pml = args.nz + 2 * args.nabc
    free_surface_start = args.nabc if args.free_surface else 1
    alpha1 = den * c * c * args.dt / args.dz
    kappa1 = case["damp"] * args.dt
    alpha2 = args.dt / (den * args.dz)
    kappa2 = torch.zeros_like(case["damp"], device=backend.device)
    kappa2[:, 1:nx_pml - 2] = 0.5 * (
        case["damp"][:, 1:nx_pml - 2] + case["damp"][:, 2:nx_pml - 1]
    ) * args.dt
    kappa3 = torch.zeros_like(case["damp"], device=backend.device)
    kappa3[free_surface_start:nz_pml - 2, :] = 0.5 * (
        case["damp"][free_surface_start:nz_pml - 2, :]
        + case["damp"][free_surface_start + 1:nz_pml - 1, :]
    ) * args.dt
    return kappa1, alpha1, kappa2, alpha2, kappa3


def run_segment(args: argparse.Namespace, backend, case, *, custom: bool) -> Dict[str, Any]:
    vp = case["vp_template"].detach().clone().requires_grad_(True)
    rho = case["rho_template"].detach().clone()
    kappa1, alpha1, kappa2, alpha2, kappa3 = build_coefficients(args, backend, case, vp, rho)
    step = step_forward_pressure_only_custom_update if custom else step_forward_pressure_only
    timer = Timer(backend)

    result, forward_seconds = timer.measure(
        lambda: step(
            args.nx,
            args.nz,
            args.dx,
            args.dz,
            args.dt,
            args.nabc,
            args.free_surface,
            case["src_x"],
            case["src_z"],
            args.shots,
            case["src_index"],
            case["src_v"],
            case["rcv_x"],
            case["rcv_z"],
            args.receivers,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            9.0 / 8.0,
            -1.0 / 24.0,
            case["p"],
            case["u"],
            case["w"],
            False,
            True,
            backend.device,
            backend.dtype,
        )
    )
    rcv_p = result[3]
    loss = rcv_p.square().sum()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    return {
        "rcv_p": rcv_p.detach(),
        "loss": loss.detach(),
        "vp_grad": vp.grad.detach(),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
    }


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, float]:
    ref = reference.detach().cpu()
    val = candidate.detach().cpu()
    diff = (val - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(ref, 1e-12))
    rel = diff / denom
    return {
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "reference_norm": float(torch.linalg.norm(ref.reshape(-1)).item()),
        "candidate_norm": float(torch.linalg.norm(val.reshape(-1)).item()),
    }


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    data = list(values)
    return {
        "min": min(data),
        "max": max(data),
        "mean": sum(data) / len(data),
        "values": data,
    }


def public_run(run: Dict[str, Any]) -> Dict[str, float]:
    return {
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": float(run["loss"].cpu().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--prefer", default="npu,cuda,cpu")
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--nt", type=int, default=120)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nabc", type=int, default=4)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--source-time", type=int, default=1)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--vp", type=float, default=2000.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--free-surface", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260605)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    backend = configure_backend(
        None if args.device == "auto" else args.device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=tuple(item.strip() for item in args.prefer.split(",") if item.strip()),
    )
    case = make_case(args, backend)

    pairs = []
    for index in range(args.warmup + args.repeat):
        reference = run_segment(args, backend, case, custom=False)
        candidate = run_segment(args, backend, case, custom=True)
        if index >= args.warmup:
            pairs.append(
                {
                    "reference": public_run(reference),
                    "candidate": public_run(candidate),
                    "comparison": {
                        "loss_abs_diff": float((reference["loss"] - candidate["loss"]).abs().cpu().item()),
                        "rcv_p": tensor_diff(reference["rcv_p"], candidate["rcv_p"]),
                        "vp_grad": tensor_diff(reference["vp_grad"], candidate["vp_grad"]),
                        "speedup": {
                            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
                            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
                            "total": reference["total_seconds"] / candidate["total_seconds"],
                        },
                        "finite": {
                            "reference_vp_grad": bool(torch.isfinite(reference["vp_grad"]).all().cpu().item()),
                            "candidate_vp_grad": bool(torch.isfinite(candidate["vp_grad"]).all().cpu().item()),
                        },
                    },
                }
            )

    result = {
        "status": "ok",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "dtype": str(args.dtype).replace("torch.", ""),
            "nx": args.nx,
            "nz": args.nz,
            "nt": args.nt,
            "nabc": args.nabc,
            "shots": args.shots,
            "receivers": args.receivers,
            "repeat": args.repeat,
            "warmup": args.warmup,
        },
        "pairs": pairs,
        "summary": {
            "speedup": {
                key: summarize(pair["comparison"]["speedup"][key] for pair in pairs)
                for key in ("forward", "backward", "total")
            },
            "max_differences": {
                "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
                "rcv_p_max_abs_diff": max(
                    pair["comparison"]["rcv_p"]["max_abs_diff"] for pair in pairs
                ),
                "vp_grad_max_abs_diff": max(
                    pair["comparison"]["vp_grad"]["max_abs_diff"] for pair in pairs
                ),
                "vp_grad_max_rel_diff": max(
                    pair["comparison"]["vp_grad"]["max_rel_diff"] for pair in pairs
                ),
            },
            "all_grad_finite": all(
                pair["comparison"]["finite"]["reference_vp_grad"]
                and pair["comparison"]["finite"]["candidate_vp_grad"]
                for pair in pairs
            ),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
