#!/usr/bin/env python
"""Benchmark tiny acoustic ADFWI forward/backward/gradient-processing runs.

The benchmark is intentionally small and in-memory so it can run on CPU, CUDA,
or NPU machines before larger case benchmarks are available. It records timing,
backend diagnostics, seed, command, git state, and model/survey dimensions as a
stable JSON report.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
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
from ADFWI.propagator import GradProcessor, TorchGradProcessor
from scripts.smoke.acoustic_backend_smoke import build_model, build_survey, parse_dtype


GRADIENT_PROCESSORS = ("none", "legacy", "torch")


def git_metadata() -> Dict[str, Any]:
    def run_git(args: list[str]) -> Optional[str]:
        try:
            proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
        except OSError:
            return None
        if proc.returncode != 0:
            return None
        return proc.stdout.strip()

    status = run_git(["status", "--short"])
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def environment_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value.detach()).cpu().item())


def synchronize(backend) -> None:
    if backend.name in ("cuda", "npu"):
        backend.synchronize()


def parse_gradient_processors(value: str) -> list[str]:
    processors = [item.strip() for item in value.split(",") if item.strip()]
    if not processors:
        raise argparse.ArgumentTypeError("at least one gradient processor is required")
    invalid = [item for item in processors if item not in GRADIENT_PROCESSORS]
    if invalid:
        supported = ", ".join(GRADIENT_PROCESSORS)
        raise argparse.ArgumentTypeError(f"unsupported gradient processor: {', '.join(invalid)}; supported: {supported}")
    return processors


def build_gradient_processor(name: str):
    if name == "legacy":
        return GradProcessor(norm_grad=False, forw_illumination=False)
    if name == "torch":
        return TorchGradProcessor(norm_grad=False, forw_illumination=False)
    if name == "none":
        return None
    raise ValueError(f"unsupported gradient processor: {name}")


def apply_gradient_processor(name: str, model, propagator) -> None:
    processor = build_gradient_processor(name)
    if processor is None:
        return
    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None before gradient processing")
    if hasattr(processor, "forward_torch"):
        model.vp.grad = processor.forward_torch(
            nx=model.nx,
            nz=model.nz,
            vmax=torch.max(model.vp.detach()),
            grad=model.vp.grad.detach(),
            forw=None,
        ).to(device=propagator.device, dtype=propagator.dtype)
        return

    processed = processor.forward(
        nx=model.nx,
        nz=model.nz,
        vmax=np.max(model.vp.detach().cpu().numpy()),
        grad=model.vp.grad.detach().cpu().numpy(),
        forw=None,
    )
    model.vp.grad = torch.as_tensor(processed, device=propagator.device, dtype=propagator.dtype)


def run_one_iteration(args: argparse.Namespace, backend, gradient_processor: str) -> Dict[str, Any]:
    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)
    from ADFWI.propagator import AcousticPropagator

    propagator = AcousticPropagator(model, survey)
    model.zero_grad(set_to_none=True)

    synchronize(backend)
    start = time.perf_counter()
    record = propagator.forward(shot_index=np.array([0]), checkpoint_segments=args.checkpoint_segments)
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
        raise RuntimeError("model.vp.grad is None after backward")
    if not torch.isfinite(pressure).all():
        raise RuntimeError("pressure contains NaN or Inf")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError("vp gradient contains NaN or Inf")

    synchronize(backend)
    start = time.perf_counter()
    apply_gradient_processor(gradient_processor, model, propagator)
    synchronize(backend)
    gradient_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after gradient processing")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError("processed vp gradient contains NaN or Inf")

    return {
        "gradient_processor": gradient_processor,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "gradient_seconds": gradient_seconds,
        "total_seconds": forward_seconds + backward_seconds + gradient_seconds,
        "loss": scalar(loss),
        "pressure_l2": tensor_norm(pressure),
        "vp_grad_norm": tensor_norm(model.vp.grad),
        "pressure_shape": list(pressure.shape),
        "pressure_dtype": str(pressure.dtype).replace("torch.", ""),
        "pressure_device": str(pressure.device),
        "memory_allocated": backend.memory_allocated(),
    }


def summarize(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def summarize_runs(runs: list[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return {
        "forward_seconds": summarize([run["forward_seconds"] for run in runs]),
        "backward_seconds": summarize([run["backward_seconds"] for run in runs]),
        "gradient_seconds": summarize([run["gradient_seconds"] for run in runs]),
        "total_seconds": summarize([run["total_seconds"] for run in runs]),
        "loss": summarize([run["loss"] for run in runs]),
        "vp_grad_norm": summarize([run["vp_grad_norm"] for run in runs]),
        "pressure_l2": summarize([run["pressure_l2"] for run in runs]),
    }


def compare_gradient_processors(runs: list[Dict[str, Any]], args: argparse.Namespace) -> list[Dict[str, Any]]:
    if "legacy" not in args.gradient_processors:
        return []
    comparisons = []
    metrics = ("loss", "vp_grad_norm", "pressure_l2")
    legacy_runs = [run for run in runs if run["gradient_processor"] == "legacy"]
    for processor in args.gradient_processors:
        if processor in {"none", "legacy"}:
            continue
        candidate_runs = [run for run in runs if run["gradient_processor"] == processor]
        pair_count = min(len(legacy_runs), len(candidate_runs))
        if pair_count == 0:
            continue
        metric_reports = []
        failed = False
        for metric in metrics:
            abs_diffs = []
            rel_diffs = []
            for idx in range(pair_count):
                reference = float(legacy_runs[idx][metric])
                value = float(candidate_runs[idx][metric])
                abs_diff = abs(value - reference)
                rel_diff = abs_diff / max(abs(reference), args.gradient_compare_atol)
                abs_diffs.append(abs_diff)
                rel_diffs.append(rel_diff)
            max_abs_diff = max(abs_diffs)
            max_rel_diff = max(rel_diffs)
            metric_failed = max_abs_diff > args.gradient_compare_atol and max_rel_diff > args.gradient_compare_rtol
            failed = failed or metric_failed
            metric_reports.append({
                "metric": metric,
                "pairs": pair_count,
                "max_abs_diff": max_abs_diff,
                "max_rel_diff": max_rel_diff,
                "status": "failed" if metric_failed else "ok",
            })
        comparisons.append({
            "reference_gradient_processor": "legacy",
            "gradient_processor": processor,
            "rtol": args.gradient_compare_rtol,
            "atol": args.gradient_compare_atol,
            "status": "failed" if failed else "ok",
            "metrics": metric_reports,
        })
    return comparisons


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())

    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    warmup_runs = []
    for processor in args.gradient_processors:
        for _ in range(args.warmup):
            warmup_runs.append(run_one_iteration(args, backend, processor))

    runs = []
    for processor in args.gradient_processors:
        for _ in range(args.repeat):
            runs.append(run_one_iteration(args, backend, processor))

    metrics_by_processor = {
        processor: summarize_runs([run for run in runs if run["gradient_processor"] == processor])
        for processor in args.gradient_processors
    }
    metrics = metrics_by_processor[args.gradient_processors[0]]
    comparisons = compare_gradient_processors(runs, args)

    report = {
        "status": "ok",
        "command": [sys.executable, *sys.argv],
        "seed": args.seed,
        "backend": backend.diagnostics(),
        "environment": environment_metadata(),
        "git": git_metadata(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "fallback_cpu": args.fallback_cpu,
            "dtype": str(args.dtype).replace("torch.", ""),
            "gradient_processors": args.gradient_processors,
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
        },
        "metrics": metrics,
        "metrics_by_gradient_processor": metrics_by_processor,
        "comparisons": comparisons,
        "runs": runs,
        "warmup_runs": warmup_runs if args.include_warmup else [],
    }
    return report


def write_report(report: Dict[str, Any], output: Optional[str]) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark tiny acoustic ADFWI forward/backward runtime.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true", help="include warmup run metrics in the JSON report")
    parser.add_argument("--gradient-processors", type=parse_gradient_processors, default=parse_gradient_processors("none"), help="comma-separated gradient processor modes: none, legacy, torch")
    parser.add_argument("--gradient-compare-rtol", type=float, default=1e-6, help="relative tolerance for legacy-vs-torch benchmark metric comparisons")
    parser.add_argument("--gradient-compare-atol", type=float, default=1e-12, help="absolute tolerance for legacy-vs-torch benchmark metric comparisons")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--nabc", type=int, default=4)
    parser.add_argument("--nt", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--f0", type=float, default=15.0)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
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
        report = run_benchmark(args)
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
