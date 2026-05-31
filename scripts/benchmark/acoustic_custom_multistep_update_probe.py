#!/usr/bin/env python
"""Probe custom-autograd acoustic multi-step recurrence.

This benchmark repeats the isolated custom p/u/w timestep update for a tiny
number of steps and compares final outputs and raw gradients against normal
PyTorch autograd. It does not modify production propagator code.
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
from scripts.benchmark.acoustic_custom_timestep_update_probe import tensor_diff, timestep_custom, timestep_reference
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_custom_multistep_update_probe_20260531.json"
)


def build_inputs(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p = torch.randn(args.shots, nz_pml, nx_pml, generator=generator, dtype=dtype) * args.pressure_scale
    u = torch.randn(args.shots, nz_pml, nx_pml - 1, generator=generator, dtype=dtype) * args.velocity_scale
    w = torch.randn(args.shots, nz_pml - 1, nx_pml, generator=generator, dtype=dtype) * args.velocity_scale
    kappa1 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * args.kappa_scale
    alpha1 = torch.ones(nz_pml, nx_pml, dtype=dtype) * args.alpha1_scale
    kappa2 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * args.kappa_scale
    alpha2 = torch.ones(nz_pml, nx_pml, dtype=dtype) * args.alpha2_scale
    kappa3 = torch.rand(nz_pml, nx_pml, generator=generator, dtype=dtype) * args.kappa_scale
    return tuple(
        tensor.to(device=device).requires_grad_(True)
        for tensor in (p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3)
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


def run_recurrence(inputs, args: argparse.Namespace, *, custom: bool):
    p, u, w, kappa1, alpha1, kappa2, alpha2, kappa3 = inputs
    update = timestep_custom if custom else timestep_reference
    for _ in range(args.steps):
        p, u, w = update(
            p,
            u,
            w,
            kappa1,
            alpha1,
            kappa2,
            alpha2,
            kappa3,
            free_surface_start=args.nabc,
        )
    return p, u, w


def run_variant(args: argparse.Namespace, backend, *, custom: bool) -> Dict[str, Any]:
    timer = Timer(backend)
    inputs = build_inputs(args, device=backend.device, dtype=backend.dtype)
    outputs, forward_seconds = timer.measure(lambda: run_recurrence(inputs, args, custom=custom))
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


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    output_names = ("p", "u", "w")
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
        "purpose": "custom autograd feasibility probe for tiny acoustic multi-step recurrence",
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
            "source_injection": False,
            "free_surface_boundary_write": False,
            "receiver_recording": False,
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
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
