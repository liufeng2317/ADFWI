#!/usr/bin/env python
"""Probe a custom-autograd acoustic pressure update.

This benchmark starts the high-impact route identified by the acoustic
performance records: reduce autograd overhead from recurrent sliced timestep
updates. It does not modify the production propagator. It compares one pressure
interior update implemented with normal PyTorch autograd against an equivalent
custom backward implementation.
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
    / "bv1.2-propagator-performance"
    / "acoustic_custom_pressure_update_probe_20260531.json"
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
    kappa = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    alpha = 1.0 + torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * 1e-3
    return tuple(t.to(device=device).requires_grad_(True) for t in (p, u, w, kappa, alpha))


def pressure_update_reference(p, u, w, kappa, alpha, *, free_surface_start: int):
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]
    z = slice(free_surface_start + 1, nz_pml - 2)
    x = slice(2, nx_pml - 2)
    result = p.clone()
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
    result[:, z, x] = (1.0 - kappa[z, x]) * p[:, z, x] - alpha[z, x] * div
    return result


class CustomPressureUpdate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, p, u, w, kappa, alpha, free_surface_start: int):
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
        result = p.clone()
        result[:, z, x] = (1.0 - kappa[z, x]) * p[:, z, x] - alpha[z, x] * div
        ctx.free_surface_start = free_surface_start
        ctx.save_for_backward(p, u, w, kappa, alpha, div)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        p, u, w, kappa, alpha, div = ctx.saved_tensors
        free_surface_start = ctx.free_surface_start
        c1 = 9.0 / 8.0
        c2 = -1.0 / 24.0
        nz_pml = p.shape[1]
        nx_pml = p.shape[2]
        z = slice(free_surface_start + 1, nz_pml - 2)
        x = slice(2, nx_pml - 2)

        grad_p = grad_output.clone()
        interior_grad = grad_output[:, z, x]
        grad_p[:, z, x] = interior_grad * (1.0 - kappa[z, x])

        grad_div = -alpha[z, x].unsqueeze(0) * interior_grad

        grad_u = torch.zeros_like(u)
        grad_u[:, z, 2 : nx_pml - 2] += c1 * grad_div
        grad_u[:, z, 1 : nx_pml - 3] -= c1 * grad_div
        grad_u[:, z, 3 : nx_pml - 1] += c2 * grad_div
        grad_u[:, z, 0 : nx_pml - 4] -= c2 * grad_div

        grad_w = torch.zeros_like(w)
        grad_w[:, z, 2 : nx_pml - 2] += c1 * grad_div
        grad_w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2] -= c1 * grad_div
        grad_w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2] += c2 * grad_div
        grad_w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2] -= c2 * grad_div

        grad_kappa = torch.zeros_like(kappa)
        grad_kappa[z, x] = torch.sum(-p[:, z, x] * interior_grad, dim=0)
        grad_alpha = torch.zeros_like(alpha)
        grad_alpha[z, x] = torch.sum(-div * interior_grad, dim=0)
        return grad_p, grad_u, grad_w, grad_kappa, grad_alpha, None


def pressure_update_custom(p, u, w, kappa, alpha, *, free_surface_start: int):
    return CustomPressureUpdate.apply(p, u, w, kappa, alpha, free_surface_start)


def run_variant(args, backend, *, custom: bool) -> Dict[str, Any]:
    timer = Timer(backend)
    inputs = build_inputs(args, device=backend.device, dtype=backend.dtype)
    update = pressure_update_custom if custom else pressure_update_reference

    output, forward_seconds = timer.measure(
        lambda: update(*inputs, free_surface_start=args.nabc)
    )
    loss = output.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    grads = [tensor.grad.detach().clone() for tensor in inputs]
    return {
        "output": output.detach(),
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
    names = ("p", "u", "w", "kappa", "alpha")
    return {
        "loss_abs_diff": abs(candidate["loss"] - reference["loss"]),
        "output": tensor_diff(reference["output"], candidate["output"]),
        "grads": {
            name: tensor_diff(ref_grad, cand_grad)
            for name, ref_grad, cand_grad in zip(names, reference["grads"], candidate["grads"])
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    data = list(values)
    return {
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(sum(data) / len(data)),
        "values": data,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )
    pairs = []
    for index in range(args.warmup + args.repeat):
        reference = run_variant(args, backend, custom=False)
        candidate = run_variant(args, backend, custom=True)
        if index >= args.warmup:
            pairs.append(
                {
                    "reference": {key: value for key, value in reference.items() if key not in {"output", "grads"}},
                    "candidate": {key: value for key, value in candidate.items() if key not in {"output", "grads"}},
                    "comparison": compare_pair(reference, candidate),
                }
            )

    return {
        "status": "ok",
        "purpose": "custom autograd feasibility probe for one acoustic pressure update",
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
        },
        "summary": {
            "speedup": {
                "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pairs),
                "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pairs),
                "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pairs),
            },
            "max_differences": {
                "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
                "output_max_abs_diff": max(pair["comparison"]["output"]["max_abs_diff"] for pair in pairs),
                "output_max_rel_diff": max(pair["comparison"]["output"]["max_rel_diff"] for pair in pairs),
                "grad_max_abs_diff": max(
                    metric["max_abs_diff"]
                    for pair in pairs
                    for metric in pair["comparison"]["grads"].values()
                ),
                "grad_max_rel_diff": max(
                    metric["max_rel_diff"]
                    for pair in pairs
                    for metric in pair["comparison"]["grads"].values()
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
