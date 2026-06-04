#!/usr/bin/env python
"""Probe a custom-autograd acoustic p/u/w timestep update.

This benchmark extends the pressure-only custom backward probe to one complete
acoustic p/u/w update without source injection or free-surface boundary writes.
It does not modify production propagator code.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_custom_timestep_update_probe_20260531.json"
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


def build_inputs(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p = torch.randn(args.shots, nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    u = torch.randn(args.shots, nz_pml, nx_pml - 1, generator=generator, dtype=dtype) * 1e-6
    w = torch.randn(args.shots, nz_pml - 1, nx_pml, generator=generator, dtype=dtype) * 1e-6
    kappa1 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    alpha1 = 1.0 + torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    kappa2 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    alpha2 = 1.0 + torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    kappa3 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    return tuple(
        tensor.to(device=device).requires_grad_(True)
        for tensor in (p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3)
    )


def timestep_reference(p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, *, free_surface_start: int):
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
        + c2 * (p_new[:, free_surface_start + 2 : nz_pml, xw] - p_new[:, free_surface_start - 1 : nz_pml - 3, xw])
    )
    w_new[:, zw, xw] = (1.0 - kappa3[zw, xw]) * w[:, zw, xw] - alpha2[zw, xw] * div_w
    return p_new, u_new, w_new


class CustomTimestepUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, free_surface_start: int):
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

        ctx.free_surface_start = free_surface_start
        ctx.save_for_backward(p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, div_p, div_u, div_w)
        return p_new, u_new, w_new

    @staticmethod
    def backward(ctx, grad_p_out, grad_u_out, grad_w_out):
        p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, div_p, div_u, div_w = ctx.saved_tensors
        free_surface_start = ctx.free_surface_start
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
        gw = grad_w_out[:, zw, xw]
        grad_w[:, zw, xw] = gw * (1.0 - kappa3[zw, xw])
        grad_kappa3[zw, xw] += torch.sum(-w[:, zw, xw] * gw, dim=0)
        grad_alpha2[zw, xw] += torch.sum(-div_w * gw, dim=0)
        grad_div_w = -alpha2[zw, xw].unsqueeze(0) * gw
        grad_p_new[:, free_surface_start + 1 : nz_pml - 1, xw] += c1 * grad_div_w
        grad_p_new[:, zw, xw] -= c1 * grad_div_w
        grad_p_new[:, free_surface_start + 2 : nz_pml, xw] += c2 * grad_div_w
        grad_p_new[:, free_surface_start - 1 : nz_pml - 3, xw] -= c2 * grad_div_w

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

        return grad_p, grad_u, grad_w, grad_kappa1, grad_alpha1, grad_kappa2, grad_alpha2, grad_kappa3, None


def timestep_custom(p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, *, free_surface_start: int):
    return CustomTimestepUpdate.apply(p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3, free_surface_start)


def run_variant(args, backend, *, custom: bool) -> Dict[str, Any]:
    timer = Timer(backend)
    inputs = build_inputs(args, device=backend.device, dtype=backend.dtype)
    update = timestep_custom if custom else timestep_reference
    outputs, forward_seconds = timer.measure(lambda: update(*inputs, free_surface_start=args.nabc))
    loss = sum(output.pow(2).mean() for output in outputs)
    _, backward_seconds = timer.measure(lambda: loss.backward())
    grads = [tensor.grad.detach().clone() for tensor in inputs]
    return {
        "outputs": [output.detach() for output in outputs],
        "loss": float(loss.detach().cpu().item()),
        "grads": grads,
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


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    output_names = ("p", "u", "w")
    grad_names = ("p", "u", "w", "kappa1", "alpha1", "kappa2", "alpha2", "kappa3")
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
        "purpose": "custom autograd feasibility probe for one acoustic p/u/w timestep update",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "source_injection": False,
            "free_surface_boundary_write": False,
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
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.shots <= 0:
        parser.error("--shots must be positive")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
