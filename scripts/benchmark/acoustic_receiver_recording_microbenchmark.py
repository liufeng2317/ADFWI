#!/usr/bin/env python
"""Compare acoustic receiver recording styles.

This Phase B benchmark isolates the differentiable receiver-output recording
pattern used in the acoustic kernel. It compares the current preallocated
``rcv[:, it, :] = sample`` style with a list-then-stack style. The production
kernel is not modified by this script.
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
from scripts.benchmark.acoustic_timestep_update_microbenchmark import (
    Timer,
    summarize,
    synchronize,
    tensor_diff,
)
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_receiver_recording_microbenchmark_20260531.json"
)


def make_case(args: argparse.Namespace, *, seed_offset: int = 0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    device = torch.device(args.resolved_device)
    dtype = args.dtype

    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    rcv_x = torch.linspace(args.nabc, args.nabc + args.nx - 1, args.receivers, device=device)
    rcv_x = rcv_x.round().to(torch.long)
    rcv_z = torch.full((args.receivers,), args.nabc + max(1, args.nz // 3), dtype=torch.long, device=device)

    shape_p = (args.nt, args.shots, nz_pml, nx_pml)
    shape_u = (args.nt, args.shots, nz_pml, nx_pml - 1)
    shape_w = (args.nt, args.shots, nz_pml - 1, nx_pml)
    p_seq = torch.randn(shape_p, device=device, dtype=dtype) * args.scale
    u_seq = torch.randn(shape_u, device=device, dtype=dtype) * args.scale
    w_seq = torch.randn(shape_w, device=device, dtype=dtype) * args.scale
    for tensor in (p_seq, u_seq, w_seq):
        tensor.requires_grad_(True)

    return {
        "p_seq": p_seq,
        "u_seq": u_seq,
        "w_seq": w_seq,
        "rcv_x": rcv_x,
        "rcv_z": rcv_z,
    }


def record_preallocated(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    device = torch.device(args.resolved_device)
    rcv_p = torch.zeros((args.shots, args.nt, args.receivers), dtype=args.dtype, device=device)
    rcv_u = torch.zeros((args.shots, args.nt, args.receivers), dtype=args.dtype, device=device)
    rcv_w = torch.zeros((args.shots, args.nt, args.receivers), dtype=args.dtype, device=device)
    rcv_x, rcv_z = state["rcv_x"], state["rcv_z"]

    for it in range(args.nt):
        rcv_p[:, it, :] = state["p_seq"][it, :, rcv_z, rcv_x]
        rcv_u[:, it, :] = state["u_seq"][it, :, rcv_z, rcv_x]
        rcv_w[:, it, :] = state["w_seq"][it, :, rcv_z, rcv_x]
    return {"p": rcv_p, "u": rcv_u, "w": rcv_w}


def record_stack(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    rcv_x, rcv_z = state["rcv_x"], state["rcv_z"]
    p_records = []
    u_records = []
    w_records = []
    for it in range(args.nt):
        p_records.append(state["p_seq"][it, :, rcv_z, rcv_x])
        u_records.append(state["u_seq"][it, :, rcv_z, rcv_x])
        w_records.append(state["w_seq"][it, :, rcv_z, rcv_x])
    return {
        "p": torch.stack(p_records, dim=1),
        "u": torch.stack(u_records, dim=1),
        "w": torch.stack(w_records, dim=1),
    }


def run_variant(args: argparse.Namespace, backend, mode: str, *, seed_offset: int) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    record_fn = record_preallocated if mode == "preallocated" else record_stack
    timer = Timer(backend)
    record, forward_seconds = timer.measure(lambda: record_fn(args, state))
    loss = record["p"].pow(2).mean() + record["u"].pow(2).mean() + record["w"].pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    gradients = {name: state[name].grad.detach().clone() for name in ("p_seq", "u_seq", "w_seq")}
    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "record": {name: value.detach().clone() for name, value in record.items()},
        "gradients": gradients,
        "finite": {
            "record_p": bool(torch.isfinite(record["p"]).all().cpu().item()),
            "record_u": bool(torch.isfinite(record["u"]).all().cpu().item()),
            "record_w": bool(torch.isfinite(record["w"]).all().cpu().item()),
            "p_grad": bool(torch.isfinite(gradients["p_seq"]).all().cpu().item()),
            "u_grad": bool(torch.isfinite(gradients["u_seq"]).all().cpu().item()),
            "w_grad": bool(torch.isfinite(gradients["w_seq"]).all().cpu().item()),
        },
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
        "records": {
            name: tensor_diff(reference["record"][name], candidate["record"][name])
            for name in ("p", "u", "w")
        },
        "gradients": {
            name: tensor_diff(reference["gradients"][name], candidate["gradients"][name])
            for name in ("p_seq", "u_seq", "w_seq")
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize_pairs(pairs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    pair_list = list(pairs)
    return {
        "speedup": {
            "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pair_list),
            "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pair_list),
            "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pair_list),
        },
        "max_differences": {
            "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pair_list),
            "p_record_max_abs_diff": max(pair["comparison"]["records"]["p"]["max_abs_diff"] for pair in pair_list),
            "u_record_max_abs_diff": max(pair["comparison"]["records"]["u"]["max_abs_diff"] for pair in pair_list),
            "w_record_max_abs_diff": max(pair["comparison"]["records"]["w"]["max_abs_diff"] for pair in pair_list),
            "p_grad_max_abs_diff": max(pair["comparison"]["gradients"]["p_seq"]["max_abs_diff"] for pair in pair_list),
            "u_grad_max_abs_diff": max(pair["comparison"]["gradients"]["u_seq"]["max_abs_diff"] for pair in pair_list),
            "w_grad_max_abs_diff": max(pair["comparison"]["gradients"]["w_seq"]["max_abs_diff"] for pair in pair_list),
        },
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
    args.resolved_device = backend.device

    warmups = []
    for index in range(args.warmup):
        reference = run_variant(args, backend, "preallocated", seed_offset=10000 + index)
        candidate = run_variant(args, backend, "stack", seed_offset=10000 + index)
        warmups.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    pairs = []
    for index in range(args.repeat):
        reference = run_variant(args, backend, "preallocated", seed_offset=index)
        candidate = run_variant(args, backend, "stack", seed_offset=index)
        pairs.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    return {
        "status": "ok",
        "purpose": "Phase B acoustic receiver recording style microbenchmark",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "shots": args.shots,
            "receivers": args.receivers,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "scale": args.scale,
        },
        "warmups": warmups if args.include_warmup else [],
        "summary": summarize_pairs(pairs),
        "pairs": pairs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--receivers", type=int, default=100)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--scale", type=float, default=1e-3)
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
