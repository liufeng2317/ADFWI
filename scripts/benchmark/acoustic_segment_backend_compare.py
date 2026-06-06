#!/usr/bin/env python
"""Compare acoustic pressure segment backends for forward parity and timing."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADFWI.backends import configure_backend
from ADFWI.propagator.acoustic_operator import (
    AcousticPressureSegmentConfig,
    AcousticPressureSegmentInputs,
    acoustic_pressure_segment,
)
from scripts.benchmark.acoustic_segment_length_probe import make_case
from scripts.smoke.acoustic_backend_smoke import parse_dtype

DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_segment_backend_compare_20260606.json"
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


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> Dict[str, Any]:
    diff = (candidate.detach() - reference.detach()).abs()
    ref_max = reference.detach().abs().max()
    return {
        "max_abs_diff": float(diff.max().detach().cpu().item()),
        "relative_max_abs_diff": float((diff.max() / torch.clamp(ref_max, min=1e-12)).detach().cpu().item()),
        "allclose_atol1e_6_rtol1e_6": bool(torch.allclose(candidate, reference, atol=1e-6, rtol=1e-6)),
        "finite_reference": bool(torch.isfinite(reference).all().detach().cpu().item()),
        "finite_candidate": bool(torch.isfinite(candidate).all().detach().cpu().item()),
    }


def build_config(args: argparse.Namespace) -> AcousticPressureSegmentConfig:
    return AcousticPressureSegmentConfig(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        dt=args.dt,
        nabc=args.nabc,
        free_surface=args.free_surface,
    )


def build_inputs(case: Dict[str, torch.Tensor], src_v: torch.Tensor) -> AcousticPressureSegmentInputs:
    return AcousticPressureSegmentInputs(
        src_x=case["src_x"],
        src_z=case["src_z"],
        src_index=case["src_index"],
        src_v=src_v,
        rcv_x=case["rcv_x"],
        rcv_z=case["rcv_z"],
        kappa1=case["kappa1"],
        alpha1=case["alpha1"],
        kappa2=case["kappa2"],
        alpha2=case["alpha2"],
        kappa3=case["kappa3"],
        p=case["p"],
        u=case["u"],
        w=case["w"],
    )


def measure(backend, fn):
    synchronize(backend)
    start = time.perf_counter()
    value = fn()
    synchronize(backend)
    return value, time.perf_counter() - start


def run_case(args: argparse.Namespace, backend, segment_nt: int) -> Dict[str, Any]:
    reference_seconds = []
    candidate_seconds = []
    compare = None
    cfg = build_config(args)
    for repeat in range(args.repeats):
        case = make_case(args, backend, segment_nt)
        inputs = build_inputs(case, case["src_v"])
        reference, ref_seconds = measure(
            backend,
            lambda: acoustic_pressure_segment(cfg, inputs, backend="torch_reference"),
        )
        candidate, cand_seconds = measure(
            backend,
            lambda: acoustic_pressure_segment(cfg, inputs, backend="custom_autograd_forward"),
        )
        reference_seconds.append(ref_seconds)
        candidate_seconds.append(cand_seconds)
        compare = {
            "rcv_p": tensor_diff(reference.rcv_p, candidate.rcv_p),
            "p": tensor_diff(reference.p, candidate.p),
            "u": tensor_diff(reference.u, candidate.u),
            "w": tensor_diff(reference.w, candidate.w),
        }

    assert compare is not None
    ref_summary = summarize(reference_seconds)
    cand_summary = summarize(candidate_seconds)
    return {
        "segment_nt": segment_nt,
        "repeats": args.repeats,
        "torch_reference_seconds": ref_summary,
        "custom_autograd_forward_seconds": cand_summary,
        "speedup_vs_torch_reference_median": ref_summary["median"] / cand_summary["median"],
        "parity_ok": all(item["allclose_atol1e_6_rtol1e_6"] for item in compare.values()),
        "finite_ok": all(item["finite_reference"] and item["finite_candidate"] for item in compare.values()),
        "diff": compare,
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
    cases = [run_case(args, backend, segment_nt) for segment_nt in args.segment_nt]
    status = "ok" if all(case["parity_ok"] and case["finite_ok"] for case in cases) else "parity_failed"
    return {
        "status": status,
        "purpose": "forward-only backend comparison for acoustic pressure segment boundary",
        "backend": {"name": backend.name, "device": str(backend.device), "dtype": str(backend.dtype)},
        "config": {
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "shots": args.shots,
            "receivers": args.receivers,
            "segment_nt": args.segment_nt,
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
    parser.add_argument("--segment-nt", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--vp", type=float, default=2200.0)
    parser.add_argument("--rho", type=float, default=2000.0)
    parser.add_argument("--source-time", type=int, default=1)
    parser.add_argument("--source-amplitude", type=float, default=1e-3)
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
                "cases": [
                    {
                        "segment_nt": case["segment_nt"],
                        "parity_ok": case["parity_ok"],
                        "finite_ok": case["finite_ok"],
                        "speedup_vs_torch_reference_median": case["speedup_vs_torch_reference_median"],
                        "rcv_p_max_abs_diff": case["diff"]["rcv_p"]["max_abs_diff"],
                        "p_max_abs_diff": case["diff"]["p"]["max_abs_diff"],
                        "u_max_abs_diff": case["diff"]["u"]["max_abs_diff"],
                        "w_max_abs_diff": case["diff"]["w"]["max_abs_diff"],
                    }
                    for case in report["cases"]
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
