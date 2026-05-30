#!/usr/bin/env python
"""Compare acoustic checkpoint vs direct-call execution for checkpoint_segments=1.

This benchmark does not change propagator code. It monkeypatches the
``ADFWI.propagator.acoustic_kernels.checkpoint`` symbol inside the current
process so that the no-checkpoint variant calls ``step_forward`` directly.

The target is a controlled performance and numerical parity experiment:

- same model/survey/backend dimensions for both variants;
- same scalar loss: mean squared pressure;
- compare all acoustic output tensors and ``vp.grad``;
- report forward/backward wall time with backend synchronization.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import BackendUnavailableError, configure_backend
from scripts.smoke.acoustic_backend_smoke import build_model, build_survey, parse_dtype


OUTPUT_KEYS = (
    "p",
    "u",
    "w",
    "forward_wavefield_p",
    "forward_wavefield_u",
    "forward_wavefield_w",
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def direct_checkpoint(function, *args, **kwargs):
    """Drop checkpoint-only kwargs and call the wrapped function directly."""

    kwargs.pop("use_reentrant", None)
    if kwargs:
        raise TypeError(f"unsupported checkpoint kwargs for direct call: {sorted(kwargs)}")
    return function(*args)


@contextmanager
def acoustic_checkpoint_mode(mode: str):
    import ADFWI.propagator.acoustic_kernels as acoustic_kernels

    original = acoustic_kernels.checkpoint
    if mode == "checkpoint":
        yield
        return
    if mode != "direct":
        raise ValueError(f"unsupported mode: {mode}")

    acoustic_kernels.checkpoint = direct_checkpoint
    try:
        yield
    finally:
        acoustic_kernels.checkpoint = original


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "device": str(detached.device),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(detached).all().cpu().item()),
        "min": float(detached.min().cpu().item()),
        "max": float(detached.max().cpu().item()),
        "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
    }


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


def run_variant(args: argparse.Namespace, backend, mode: str) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)

    from ADFWI.propagator import AcousticPropagator

    propagator = AcousticPropagator(model, survey)
    model.zero_grad(set_to_none=True)

    with acoustic_checkpoint_mode(mode):
        synchronize(backend)
        start = time.perf_counter()
        record = propagator.forward(
            shot_index=np.array([0]),
            checkpoint_segments=args.checkpoint_segments,
        )
        synchronize(backend)
        forward_seconds = time.perf_counter() - start

        pressure = record["p"]
        loss = pressure.pow(2).mean()

        synchronize(backend)
        start = time.perf_counter()
        loss.backward()
        synchronize(backend)
        backward_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError(f"{mode}: model.vp.grad is None after backward")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError(f"{mode}: model.vp.grad contains NaN or Inf")

    outputs = {key: record[key].detach().clone() for key in OUTPUT_KEYS}
    grad = model.vp.grad.detach().clone()

    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": float(loss.detach().cpu().item()),
        "outputs": outputs,
        "vp_grad": grad,
        "summaries": {key: tensor_summary(value) for key, value in outputs.items()},
        "vp_grad_summary": tensor_summary(grad),
        "memory_allocated": backend.memory_allocated(),
    }


def public_run_payload(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": run["mode"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": run["loss"],
        "summaries": run["summaries"],
        "vp_grad_summary": run["vp_grad_summary"],
        "memory_allocated": run["memory_allocated"],
    }


def compare_runs(reference: Dict[str, Any], candidate: Dict[str, Any], *, atol_floor: float) -> Dict[str, Any]:
    output_diffs = {
        key: tensor_diff(reference["outputs"][key], candidate["outputs"][key], atol_floor=atol_floor)
        for key in OUTPUT_KEYS
    }
    grad_diff = tensor_diff(reference["vp_grad"], candidate["vp_grad"], atol_floor=atol_floor)
    loss_abs_diff = abs(candidate["loss"] - reference["loss"])
    loss_rel_diff = loss_abs_diff / max(abs(reference["loss"]), atol_floor)
    return {
        "reference": reference["mode"],
        "candidate": candidate["mode"],
        "loss_abs_diff": loss_abs_diff,
        "loss_rel_diff": loss_rel_diff,
        "output_diffs": output_diffs,
        "vp_grad_diff": grad_diff,
        "forward_speedup": reference["forward_seconds"] / candidate["forward_seconds"]
        if candidate["forward_seconds"] > 0
        else None,
        "backward_speedup": reference["backward_seconds"] / candidate["backward_seconds"]
        if candidate["backward_seconds"] > 0
        else None,
        "total_speedup": reference["total_seconds"] / candidate["total_seconds"]
        if candidate["total_seconds"] > 0
        else None,
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(sum(data) / len(data)),
    }


def summarize_mode(runs: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return {
        "forward_seconds": summarize(run["forward_seconds"] for run in runs),
        "backward_seconds": summarize(run["backward_seconds"] for run in runs),
        "total_seconds": summarize(run["total_seconds"] for run in runs),
        "loss": summarize(run["loss"] for run in runs),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    if args.checkpoint_segments != 1:
        raise ValueError("this experiment is only valid for checkpoint_segments=1")

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
        warmups.append(public_run_payload(run_variant(args, backend, "checkpoint")))
        warmups.append(public_run_payload(run_variant(args, backend, "direct")))

    pairs = []
    checkpoint_runs = []
    direct_runs = []
    for _ in range(args.repeat):
        checkpoint_run = run_variant(args, backend, "checkpoint")
        direct_run = run_variant(args, backend, "direct")
        checkpoint_runs.append(public_run_payload(checkpoint_run))
        direct_runs.append(public_run_payload(direct_run))
        pairs.append(
            {
                "checkpoint": public_run_payload(checkpoint_run),
                "direct": public_run_payload(direct_run),
                "comparison": compare_runs(checkpoint_run, direct_run, atol_floor=args.atol_floor),
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
        "summary": {
            "checkpoint": summarize_mode(checkpoint_runs),
            "direct": summarize_mode(direct_runs),
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

