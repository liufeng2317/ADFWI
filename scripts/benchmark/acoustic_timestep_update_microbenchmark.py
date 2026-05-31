#!/usr/bin/env python
"""Compare acoustic timestep update styles for autograd overhead.

This is a bounded Phase B microbenchmark. It does not call or modify the
production acoustic kernel. It isolates the pressure update pattern that the
operator profile identified as a major source of SliceBackward/CopySlices
overhead, then compares:

- current-style cloned tensor with sliced assignment;
- functional reconstruction with ``torch.cat``.

The goal is to decide whether a real kernel rewrite is worth investigating.
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
    / "acoustic_timestep_update_microbenchmark_20260531.json"
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


def make_case(args: argparse.Namespace, *, seed_offset: int = 0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    device = torch.device(args.resolved_device)
    dtype = args.dtype

    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p = torch.randn((args.shots, nz_pml, nx_pml), device=device, dtype=dtype) * args.scale
    u = torch.randn((args.shots, nz_pml, nx_pml - 1), device=device, dtype=dtype) * args.scale
    w = torch.randn((args.shots, nz_pml - 1, nx_pml), device=device, dtype=dtype) * args.scale
    kappa1 = torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.01
    alpha1 = torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.02

    for tensor in (p, u, w):
        tensor.requires_grad_(True)

    return {
        "p": p,
        "u": u,
        "w": w,
        "kappa1": kappa1,
        "alpha1": alpha1,
    }


def pressure_inner_update(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    return (
        (1.0 - kappa1[z0:z1, x0:x1]) * p[:, z0:z1, x0:x1]
        - alpha1[z0:z1, x0:x1]
        * (
            c1
            * (
                u[:, z0:z1, x0:x1]
                - u[:, z0:z1, x0 - 1:x1 - 1]
                + w[:, z0:z1, x0:x1]
                - w[:, z0 - 1:z1 - 1, x0:x1]
            )
            + c2
            * (
                u[:, z0:z1, x0 + 1:x1 + 1]
                - u[:, z0:z1, x0 - 2:x1 - 2]
                + w[:, z0 + 1:z1 + 1, x0:x1]
                - w[:, z0 - 2:z1 - 2, x0:x1]
            )
        )
    )


def update_sliced_assignment(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    p_next = state["p"].clone()
    p_next[:, z0:z1, x0:x1] = pressure_inner_update(args, state)
    return p_next


def update_functional_reconstruct(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    p = state["p"]
    inner = pressure_inner_update(args, state)
    middle = torch.cat((p[:, z0:z1, :x0], inner, p[:, z0:z1, x1:]), dim=2)
    return torch.cat((p[:, :z0, :], middle, p[:, z1:, :]), dim=1)


def run_variant(args: argparse.Namespace, backend, mode: str, *, seed_offset: int = 0) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    update_fn = update_sliced_assignment if mode == "sliced_assignment" else update_functional_reconstruct
    timer = Timer(backend)
    p_next, forward_seconds = timer.measure(lambda: update_fn(args, state))
    loss = p_next.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    gradients = {name: state[name].grad.detach().clone() for name in ("p", "u", "w")}
    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "output": p_next.detach().clone(),
        "gradients": gradients,
        "finite": {
            "output": bool(torch.isfinite(p_next).all().cpu().item()),
            "p_grad": bool(torch.isfinite(gradients["p"]).all().cpu().item()),
            "u_grad": bool(torch.isfinite(gradients["u"]).all().cpu().item()),
            "w_grad": bool(torch.isfinite(gradients["w"]).all().cpu().item()),
        },
    }


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor, *, atol_floor=1e-12) -> Dict[str, Any]:
    ref = reference.detach().cpu()
    val = candidate.detach().cpu()
    diff = (val - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(ref, atol_floor))
    rel = diff / denom
    return {
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "reference_norm": float(torch.linalg.norm(ref.reshape(-1)).item()),
        "candidate_norm": float(torch.linalg.norm(val.reshape(-1)).item()),
    }


def public_variant(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": run["mode"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": float(run["loss"].cpu().item()),
        "finite": run["finite"],
    }


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss_abs_diff": float((candidate["loss"].cpu() - reference["loss"].cpu()).abs().item()),
        "output": tensor_diff(reference["output"], candidate["output"]),
        "gradients": {
            name: tensor_diff(reference["gradients"][name], candidate["gradients"][name])
            for name in ("p", "u", "w")
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def event_payload(event, *, index: int) -> Dict[str, Any]:
    payload = {
        "rank": index,
        "key": event.key,
        "count": int(event.count),
        "cpu_time_self_us": float(event.self_cpu_time_total),
        "cpu_time_total_us": float(event.cpu_time_total),
    }
    for attr in ("device_time_total", "self_device_time_total"):
        if hasattr(event, attr):
            payload[f"{attr}_us"] = float(getattr(event, attr))
    return payload


def profile_variant(args: argparse.Namespace, backend, mode: str) -> Dict[str, Any]:
    state = make_case(args, seed_offset=1000)
    update_fn = update_sliced_assignment if mode == "sliced_assignment" else update_functional_reconstruct
    p_next = update_fn(args, state)
    loss = p_next.pow(2).mean()
    synchronize(backend)
    use_device = "npu" if backend.name == "npu" else None
    with torch.autograd.profiler.profile(use_device=use_device, use_cpu=True) as prof:
        start = time.perf_counter()
        loss.backward()
        synchronize(backend)
        backward_seconds = time.perf_counter() - start
    sort_key = "self_device_time_total" if backend.name == "npu" else "self_cpu_time_total"
    events = sorted(prof.key_averages(), key=lambda event: getattr(event, sort_key, 0.0), reverse=True)
    return {
        "mode": mode,
        "backward_seconds": backward_seconds,
        "top_events_sort": sort_key,
        "top_events": [event_payload(event, index=index + 1) for index, event in enumerate(events[: args.topk])],
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {"min": min(data), "max": max(data), "mean": sum(data) / len(data), "values": data}


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )
    args.resolved_device = backend.device

    warmups = []
    for index in range(args.warmup):
        reference = run_variant(args, backend, "sliced_assignment", seed_offset=10000 + index)
        candidate = run_variant(args, backend, "functional_reconstruct", seed_offset=10000 + index)
        warmups.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    pairs = []
    for index in range(args.repeat):
        reference = run_variant(args, backend, "sliced_assignment", seed_offset=index)
        candidate = run_variant(args, backend, "functional_reconstruct", seed_offset=index)
        pairs.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    profiles = []
    if args.profile:
        profiles = [
            profile_variant(args, backend, "sliced_assignment"),
            profile_variant(args, backend, "functional_reconstruct"),
        ]

    return {
        "status": "ok",
        "purpose": "Phase B acoustic pressure update style microbenchmark",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "scale": args.scale,
            "free_surface": args.free_surface,
            "profile": args.profile,
            "topk": args.topk,
        },
        "warmups": warmups if args.include_warmup else [],
        "summary": {
            "speedup": {
                "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pairs),
                "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pairs),
                "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pairs),
            },
            "max_differences": {
                "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
                "output_max_abs_diff": max(pair["comparison"]["output"]["max_abs_diff"] for pair in pairs),
                "p_grad_max_abs_diff": max(pair["comparison"]["gradients"]["p"]["max_abs_diff"] for pair in pairs),
                "u_grad_max_abs_diff": max(pair["comparison"]["gradients"]["u"]["max_abs_diff"] for pair in pairs),
                "w_grad_max_abs_diff": max(pair["comparison"]["gradients"]["w"]["max_abs_diff"] for pair in pairs),
            },
        },
        "pairs": pairs,
        "profiles": profiles,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topk", type=int, default=12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
