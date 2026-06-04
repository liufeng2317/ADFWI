#!/usr/bin/env python
"""Sweep acoustic checkpoint segment counts for time and numerical parity.

This benchmark does not change propagator code. It compares the production
``checkpoint_segments`` path against ``checkpoint_segments=1`` on the same
small acoustic case.
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
from ADFWI.propagator import AcousticPropagator
from scripts.smoke.acoustic_backend_smoke import build_model, build_survey, parse_dtype


OUTPUT_KEYS = (
    "p",
    "u",
    "w",
    "forward_wavefield_p",
    "forward_wavefield_u",
    "forward_wavefield_w",
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_checkpoint_segment_sweep_20260531.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


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


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "finite": bool(torch.isfinite(detached).all().cpu().item()),
        "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
    }


def run_variant(args: argparse.Namespace, backend, *, checkpoint_segments: int, seed: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)
    propagator = AcousticPropagator(model, survey)
    model.zero_grad(set_to_none=True)

    synchronize(backend)
    start = time.perf_counter()
    record = propagator.forward(
        shot_index=np.array([0]),
        checkpoint_segments=checkpoint_segments,
        save_forward_wavefield=args.save_forward_wavefield,
    )
    synchronize(backend)
    forward_seconds = time.perf_counter() - start

    loss = record[args.loss_component].pow(2).mean()
    synchronize(backend)
    start = time.perf_counter()
    loss.backward()
    synchronize(backend)
    backward_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError(f"segments={checkpoint_segments}: model.vp.grad is None")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError(f"segments={checkpoint_segments}: model.vp.grad contains NaN or Inf")

    return {
        "checkpoint_segments": checkpoint_segments,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": float(loss.detach().cpu().item()),
        "outputs": {key: record[key].detach().clone() for key in OUTPUT_KEYS},
        "vp_grad": model.vp.grad.detach().clone(),
        "memory_allocated": backend.memory_allocated(),
    }


def public_variant(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "checkpoint_segments": run["checkpoint_segments"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": run["loss"],
        "outputs": {key: tensor_summary(value) for key, value in run["outputs"].items()},
        "vp_grad": tensor_summary(run["vp_grad"]),
        "memory_allocated": run["memory_allocated"],
    }


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss_abs_diff": abs(candidate["loss"] - reference["loss"]),
        "outputs": {
            key: tensor_diff(reference["outputs"][key], candidate["outputs"][key])
            for key in OUTPUT_KEYS
        },
        "vp_grad": tensor_diff(reference["vp_grad"], candidate["vp_grad"]),
        "speedup_vs_segments_1": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {"min": min(data), "max": max(data), "mean": sum(data) / len(data), "values": data}


def parse_segments(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("at least one segment count is required")
    if any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("segment counts must be positive")
    return result


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )

    if 1 not in args.segments:
        args.segments = [1, *args.segments]

    warmups = []
    for index in range(args.warmup):
        warmups.append(public_variant(run_variant(args, backend, checkpoint_segments=1, seed=args.seed + 10000 + index)))

    repeats = []
    for repeat in range(args.repeat):
        seed = args.seed + repeat
        reference = run_variant(args, backend, checkpoint_segments=1, seed=seed)
        variants = []
        for segments in args.segments:
            if segments == 1:
                run = reference
                comparison = None
            else:
                run = run_variant(args, backend, checkpoint_segments=segments, seed=seed)
                comparison = compare(reference, run)
            variants.append({"run": public_variant(run), "comparison": comparison})
        repeats.append({"repeat": repeat, "variants": variants})

    summary = {}
    for segments in args.segments:
        runs = [
            variant["run"]
            for repeat in repeats
            for variant in repeat["variants"]
            if variant["run"]["checkpoint_segments"] == segments
        ]
        summary[str(segments)] = {
            "forward_seconds": summarize(run["forward_seconds"] for run in runs),
            "backward_seconds": summarize(run["backward_seconds"] for run in runs),
            "total_seconds": summarize(run["total_seconds"] for run in runs),
        }

    max_differences = {}
    for segments in args.segments:
        if segments == 1:
            continue
        comparisons = [
            variant["comparison"]
            for repeat in repeats
            for variant in repeat["variants"]
            if variant["run"]["checkpoint_segments"] == segments
        ]
        max_differences[str(segments)] = {
            "loss_abs_diff": max(item["loss_abs_diff"] for item in comparisons),
            "vp_grad_max_abs_diff": max(item["vp_grad"]["max_abs_diff"] for item in comparisons),
            "vp_grad_max_rel_diff": max(item["vp_grad"]["max_rel_diff"] for item in comparisons),
            "p_max_abs_diff": max(item["outputs"]["p"]["max_abs_diff"] for item in comparisons),
            "p_max_rel_diff": max(item["outputs"]["p"]["max_rel_diff"] for item in comparisons),
        }

    return {
        "status": "ok",
        "purpose": "Phase B acoustic checkpoint segment sweep",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "segments": args.segments,
            "save_forward_wavefield": args.save_forward_wavefield,
            "loss_component": args.loss_component,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "dx": args.dx,
            "dz": args.dz,
            "dt": args.dt,
            "f0": args.f0,
        },
        "summary": summary,
        "max_differences": max_differences,
        "warmups": warmups if args.include_warmup else [],
        "repeats": repeats,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--segments", type=parse_segments, default=parse_segments("1,2,4,8"))
    parser.add_argument("--save-forward-wavefield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--loss-component", choices=("p", "u", "w"), default="p")
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--dx", type=float, default=40.0)
    parser.add_argument("--dz", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--f0", type=float, default=5.0)
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
