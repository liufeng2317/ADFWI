#!/usr/bin/env python
"""Check acoustic short-segment parity against a full pressure-only run.

The future fused multi-time-step route needs an exact boundary contract before
any compiled or custom-autograd implementation is useful. This probe compares:

- one full production ``step_forward_pressure_only`` call over ``nt``;
- repeated production calls over shorter ``segment_nt`` chunks.

It compares receiver output and final boundary state ``p/u/w``. Production
propagator code is not modified.
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
from ADFWI.propagator.acoustic_kernels import step_forward_pressure_only
from scripts.benchmark.acoustic_segment_length_probe import make_case
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_segment_contract_probe_20260606.json"
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
    ref = reference.detach()
    val = candidate.detach()
    diff = (val - ref).abs()
    ref_max = ref.abs().max()
    return {
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max().detach().cpu().item()),
        "relative_max_abs_diff": float((diff.max() / torch.clamp(ref_max, min=1e-12)).detach().cpu().item()),
        "reference_abs_max": float(ref_max.detach().cpu().item()),
        "candidate_abs_max": float(val.abs().max().detach().cpu().item()),
        "allclose_atol1e_6_rtol1e_6": bool(torch.allclose(val, ref, atol=1e-6, rtol=1e-6)),
        "finite_reference": bool(torch.isfinite(ref).all().detach().cpu().item()),
        "finite_candidate": bool(torch.isfinite(val).all().detach().cpu().item()),
    }


def run_step(args: argparse.Namespace, backend, case: Dict[str, torch.Tensor], src_v: torch.Tensor):
    return step_forward_pressure_only(
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
        src_v,
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


def run_full(args: argparse.Namespace, backend):
    case = make_case(args, backend, args.nt)
    return run_step(args, backend, case, case["src_v"])


def run_chunked(args: argparse.Namespace, backend, segment_nt: int):
    case = make_case(args, backend, args.nt)
    rcv_chunks: List[torch.Tensor] = []
    p = case["p"]
    u = case["u"]
    w = case["w"]
    for start in range(0, args.nt, segment_nt):
        end = min(start + segment_nt, args.nt)
        case["p"] = p
        case["u"] = u
        case["w"] = w
        p, u, w, rcv_p, _ = run_step(args, backend, case, case["src_v"][:, start:end])
        rcv_chunks.append(rcv_p)
    return p, u, w, torch.cat(rcv_chunks, dim=1), None


def measure(backend, fn):
    synchronize(backend)
    start = time.perf_counter()
    value = fn()
    synchronize(backend)
    return value, time.perf_counter() - start


def run_case(args: argparse.Namespace, backend, segment_nt: int) -> Dict[str, Any]:
    full_times: List[float] = []
    chunk_times: List[float] = []
    final_compare: Optional[Dict[str, Any]] = None
    for _ in range(args.repeats):
        full_result, full_seconds = measure(backend, lambda: run_full(args, backend))
        chunk_result, chunk_seconds = measure(backend, lambda: run_chunked(args, backend, segment_nt))
        full_times.append(full_seconds)
        chunk_times.append(chunk_seconds)
        full_p, full_u, full_w, full_rcv_p, _ = full_result
        chunk_p, chunk_u, chunk_w, chunk_rcv_p, _ = chunk_result
        final_compare = {
            "rcv_p": tensor_diff(full_rcv_p, chunk_rcv_p),
            "p": tensor_diff(full_p, chunk_p),
            "u": tensor_diff(full_u, chunk_u),
            "w": tensor_diff(full_w, chunk_w),
        }

    assert final_compare is not None
    parity_ok = all(item["allclose_atol1e_6_rtol1e_6"] for item in final_compare.values())
    finite_ok = all(
        item["finite_reference"] and item["finite_candidate"]
        for item in final_compare.values()
    )
    full_summary = summarize(full_times)
    chunk_summary = summarize(chunk_times)
    return {
        "segment_nt": segment_nt,
        "repeats": args.repeats,
        "full_seconds": full_summary,
        "chunked_seconds": chunk_summary,
        "chunked_over_full_median": (
            chunk_summary["median"] / full_summary["median"]
            if full_summary["median"] > 0
            else None
        ),
        "parity_ok": parity_ok,
        "finite_ok": finite_ok,
        "diff": final_compare,
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
        "purpose": "short-segment pressure-only contract for future fused multi-time-step work",
        "backend": {"name": backend.name, "device": str(backend.device), "dtype": str(backend.dtype)},
        "config": {
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "shots": args.shots,
            "receivers": args.receivers,
            "nt": args.nt,
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
    parser.add_argument("--nt", type=int, default=16)
    parser.add_argument("--segment-nt", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--repeats", type=int, default=3)
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
                "config": report["config"],
                "cases": [
                    {
                        "segment_nt": case["segment_nt"],
                        "parity_ok": case["parity_ok"],
                        "finite_ok": case["finite_ok"],
                        "chunked_over_full_median": case["chunked_over_full_median"],
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
