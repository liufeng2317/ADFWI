#!/usr/bin/env python
"""Probe pressure-only acoustic segment cost as time length increases.

This script estimates whether a future fused multi-time-step segment is worth
building. It runs the current production ``step_forward_pressure_only`` path
with several ``nt`` values and reports total time and seconds per time step.

The script does not modify production propagator code.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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
    / "acoustic_segment_length_probe_20260606.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    data = list(values)
    return {
        "mean": float(statistics.mean(data)),
        "median": float(statistics.median(data)),
        "min": float(min(data)),
        "max": float(max(data)),
        "values": data,
    }


def make_case(args: argparse.Namespace, backend, nt: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + nt)
    device = backend.device
    dtype = backend.dtype
    nx_pml = args.nx + 2 * args.nabc
    nz_pml = args.nz + 2 * args.nabc

    vp = torch.full((args.nz, args.nx), args.vp, dtype=dtype, device=device)
    rho = torch.full((args.nz, args.nx), args.rho, dtype=dtype, device=device)
    damp = torch.zeros((nz_pml, nx_pml), dtype=dtype, device=device)

    src_x = torch.linspace(2, args.nx - 3, args.shots, device=device).round().long() + args.nabc
    src_z = torch.full((args.shots,), 2 + args.nabc, dtype=torch.long, device=device)
    rcv_x = torch.linspace(1, args.nx - 2, args.receivers, device=device).round().long() + args.nabc
    rcv_z = torch.full((args.receivers,), 2 + args.nabc, dtype=torch.long, device=device)
    src_index = torch.arange(args.shots, dtype=torch.long, device=device)

    src_v = torch.zeros((args.shots, nt), dtype=dtype, device=device)
    source_time = min(max(args.source_time, 0), nt - 1)
    src_v[:, source_time] = args.source_amplitude

    p = torch.zeros((args.shots, nz_pml, nx_pml), dtype=dtype, device=device)
    u = torch.zeros((args.shots, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((args.shots, nz_pml - 1, nx_pml), dtype=dtype, device=device)

    c = pad_torchSingle(vp, args.nabc, args.nz, args.nx, args.shots, device=device)
    den = pad_torchSingle(rho, args.nabc, args.nz, args.nx, args.shots, device=device)
    kappa1 = damp * args.dt
    alpha1 = den * c.pow(2) * args.dt
    kappa2 = damp[:, 1:] * args.dt
    alpha2 = 1.0 / den[:, 1:] * args.dt
    kappa3 = damp[1:, :] * args.dt
    c1_staggered = 9.0 / 8.0
    c2_staggered = -1.0 / 24.0

    return {
        "src_x": src_x,
        "src_z": src_z,
        "src_index": src_index,
        "src_v": src_v,
        "rcv_x": rcv_x,
        "rcv_z": rcv_z,
        "kappa1": kappa1,
        "alpha1": alpha1,
        "kappa2": kappa2,
        "alpha2": alpha2,
        "kappa3": kappa3,
        "c1_staggered": torch.as_tensor(c1_staggered, dtype=dtype, device=device),
        "c2_staggered": torch.as_tensor(c2_staggered, dtype=dtype, device=device),
        "p": p,
        "u": u,
        "w": w,
    }


def run_segment(args: argparse.Namespace, backend, nt: int) -> Dict[str, Any]:
    seconds: List[float] = []
    output_summary: Optional[Dict[str, Any]] = None
    for repeat in range(args.repeats):
        case = make_case(args, backend, nt)
        synchronize(backend)
        start = time.perf_counter()
        p, u, w, rcv_p, forward_wavefield_p = step_forward_pressure_only(
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
            case["kappa1"],
            case["alpha1"],
            case["kappa2"],
            case["alpha2"],
            case["kappa3"],
            float(case["c1_staggered"].item()),
            float(case["c2_staggered"].item()),
            case["p"],
            case["u"],
            case["w"],
            save_forward_wavefield=False,
            accumulate_wavefield_in_grad=False,
            device=backend.device,
            dtype=backend.dtype,
        )
        synchronize(backend)
        elapsed = time.perf_counter() - start
        seconds.append(elapsed)
        if repeat == args.repeats - 1:
            output_summary = {
                "rcv_p_shape": list(rcv_p.shape),
                "rcv_p_finite": bool(torch.isfinite(rcv_p).all().detach().cpu().item()),
                "rcv_p_abs_max": float(rcv_p.abs().max().detach().cpu().item()),
                "p_abs_max": float(p.abs().max().detach().cpu().item()),
                "u_abs_max": float(u.abs().max().detach().cpu().item()),
                "w_abs_max": float(w.abs().max().detach().cpu().item()),
            }

    total = summarize(seconds)
    seconds_per_step = [value / nt for value in seconds]
    return {
        "nt": nt,
        "repeats": args.repeats,
        "total_seconds": total,
        "seconds_per_step": summarize(seconds_per_step),
        "output": output_summary,
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )
    cases = [run_segment(args, backend, nt) for nt in args.nt]
    baseline = next((case for case in cases if case["nt"] == min(args.nt)), cases[0])
    baseline_per_step = baseline["seconds_per_step"]["median"]
    for case in cases:
        case["per_step_speedup_vs_min_nt"] = (
            baseline_per_step / case["seconds_per_step"]["median"]
            if case["seconds_per_step"]["median"] > 0
            else None
        )
    return {
        "status": "ok",
        "purpose": "production pressure-only segment length probe for future fused multi-time-step design",
        "backend": {"name": backend.name, "device": str(backend.device), "dtype": str(backend.dtype)},
        "config": {
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "shots": args.shots,
            "receivers": args.receivers,
            "nt": args.nt,
            "repeats": args.repeats,
            "free_surface": args.free_surface,
        },
        "cases": cases,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefer", default="npu,cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", type=parse_dtype, default=torch.float32)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nabc", type=int, default=8)
    parser.add_argument("--shots", type=int, default=40)
    parser.add_argument("--receivers", type=int, default=64)
    parser.add_argument("--nt", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vp", type=float, default=2200.0)
    parser.add_argument("--rho", type=float, default=2000.0)
    parser.add_argument("--source-time", type=int, default=1)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--free-surface", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "backend": report["backend"],
                "config": report["config"],
                "cases": [
                    {
                        "nt": case["nt"],
                        "median_seconds": case["total_seconds"]["median"],
                        "median_seconds_per_step": case["seconds_per_step"]["median"],
                        "per_step_speedup_vs_min_nt": case["per_step_speedup_vs_min_nt"],
                        "finite": case["output"]["rcv_p_finite"],
                    }
                    for case in report["cases"]
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
