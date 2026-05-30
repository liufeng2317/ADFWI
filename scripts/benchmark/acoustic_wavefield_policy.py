#!/usr/bin/env python
"""Compare acoustic forward-wavefield output policy against receiver gradients.

This benchmark checks the safety boundary for ``save_forward_wavefield=False``:
receiver records, scalar pressure loss, and ``vp.grad`` must remain identical to
the default path. Detached ``forward_wavefield_*`` outputs are expected to be
zero in the skipped mode and are not used for this gradient comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import BackendUnavailableError, configure_backend
from ADFWI.propagator import AcousticPropagator
from scripts.smoke.acoustic_backend_smoke import build_model, build_survey, parse_dtype


RECEIVER_KEYS = ("p", "u", "w")
WAVEFIELD_KEYS = ("forward_wavefield_p", "forward_wavefield_u", "forward_wavefield_w")


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor, *, atol_floor: float) -> Dict[str, Any]:
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


def run_variant(args: argparse.Namespace, backend, *, save_forward_wavefield: bool) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)
    propagator = AcousticPropagator(model, survey)
    model.zero_grad(set_to_none=True)

    synchronize(backend)
    start = time.perf_counter()
    record = propagator.forward(
        shot_index=np.array([0]),
        checkpoint_segments=args.checkpoint_segments,
        save_forward_wavefield=save_forward_wavefield,
    )
    synchronize(backend)
    forward_seconds = time.perf_counter() - start

    loss = record["p"].pow(2).mean()

    synchronize(backend)
    start = time.perf_counter()
    loss.backward()
    synchronize(backend)
    backward_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after backward")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError("model.vp.grad contains NaN or Inf")

    return {
        "save_forward_wavefield": save_forward_wavefield,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "record": {key: value.detach().clone() for key, value in record.items()},
        "vp_grad": model.vp.grad.detach().clone(),
        "memory_allocated": backend.memory_allocated(),
    }


def public_variant(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "save_forward_wavefield": run["save_forward_wavefield"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": float(run["loss"].cpu().item()),
        "memory_allocated": run["memory_allocated"],
        "wavefield_norms": {
            key: float(torch.linalg.norm(run["record"][key].detach().cpu().reshape(-1)).item())
            for key in WAVEFIELD_KEYS
        },
    }


def compare_runs(full: Dict[str, Any], skipped: Dict[str, Any], *, atol_floor: float) -> Dict[str, Any]:
    loss_abs_diff = float((skipped["loss"].detach().cpu() - full["loss"].detach().cpu()).abs().item())
    loss_rel_diff = loss_abs_diff / max(abs(float(full["loss"].detach().cpu().item())), atol_floor)
    return {
        "reference": "save_forward_wavefield=True",
        "candidate": "save_forward_wavefield=False",
        "loss_abs_diff": loss_abs_diff,
        "loss_rel_diff": loss_rel_diff,
        "receiver_diffs": {
            key: tensor_diff(full["record"][key], skipped["record"][key], atol_floor=atol_floor)
            for key in RECEIVER_KEYS
        },
        "vp_grad_diff": tensor_diff(full["vp_grad"], skipped["vp_grad"], atol_floor=atol_floor),
        "wavefield_norms": {
            key: {
                "full": float(torch.linalg.norm(full["record"][key].detach().cpu().reshape(-1)).item()),
                "skipped": float(torch.linalg.norm(skipped["record"][key].detach().cpu().reshape(-1)).item()),
            }
            for key in WAVEFIELD_KEYS
        },
        "forward_speedup": full["forward_seconds"] / skipped["forward_seconds"]
        if skipped["forward_seconds"] > 0
        else None,
        "backward_speedup": full["backward_seconds"] / skipped["backward_seconds"]
        if skipped["backward_seconds"] > 0
        else None,
        "total_speedup": full["total_seconds"] / skipped["total_seconds"]
        if skipped["total_seconds"] > 0
        else None,
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.checkpoint_segments != 1:
        raise ValueError("this benchmark currently expects checkpoint_segments=1")

    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )

    warmups = []
    for _ in range(args.warmup):
        warmups.append(public_variant(run_variant(args, backend, save_forward_wavefield=True)))
        warmups.append(public_variant(run_variant(args, backend, save_forward_wavefield=False)))

    pairs = []
    for _ in range(args.repeat):
        full = run_variant(args, backend, save_forward_wavefield=True)
        skipped = run_variant(args, backend, save_forward_wavefield=False)
        comparison = compare_runs(full, skipped, atol_floor=args.atol_floor)
        pairs.append(
            {
                "full": public_variant(full),
                "skipped": public_variant(skipped),
                "comparison": comparison,
            }
        )

    return {
        "status": "ok",
        "command": [sys.executable, *sys.argv],
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "fallback_cpu": args.fallback_cpu,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "checkpoint_segments": args.checkpoint_segments,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "dt": args.dt,
            "f0": args.f0,
            "dx": args.dx,
            "dz": args.dz,
            "atol_floor": args.atol_floor,
        },
        "pairs": pairs,
        "warmups": warmups if args.include_warmup else [],
    }


def write_report(report: Dict[str, Any], output: Optional[str]) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--f0", type=float, default=5.0)
    parser.add_argument("--dx", type=float, default=40.0)
    parser.add_argument("--dz", type=float, default=40.0)
    parser.add_argument("--atol-floor", type=float, default=1e-12)
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    try:
        report = run_experiment(args)
    except BackendUnavailableError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1

    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
