#!/usr/bin/env python
"""Benchmark legacy versus torch-native gradient processor paths."""

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
from scripts.smoke.acoustic_backend_smoke import parse_dtype


PROCESSOR_CASES = ("norm", "marine_smooth", "land_smooth", "illumination", "mask_illumination")
STRICT_RTOL = 1e-5
STRICT_ATOL = 2e-3
TOLERANCE_PROFILES = {
    "strict": {"rtol": STRICT_RTOL, "atol": STRICT_ATOL},
    "npu-float32": {"rtol": 2e-4, "atol": 5e-1},
}


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


def synchronize(backend) -> None:
    if backend.name in ("cuda", "npu"):
        backend.synchronize()


def parse_case_list(value: str) -> list[str]:
    cases = [item.strip() for item in value.split(",") if item.strip()]
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    invalid = [item for item in cases if item not in PROCESSOR_CASES]
    if invalid:
        supported = ", ".join(PROCESSOR_CASES)
        raise argparse.ArgumentTypeError(f"unsupported case: {', '.join(invalid)}; supported: {supported}")
    return cases


def summarize(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
    }


def parse_tolerance_profile(value: str) -> str:
    if value not in TOLERANCE_PROFILES:
        supported = ", ".join(sorted(TOLERANCE_PROFILES))
        raise argparse.ArgumentTypeError(f"unsupported tolerance profile: {value}; supported: {supported}")
    return value


def resolve_tolerances(args: argparse.Namespace) -> Tuple[float, float]:
    profile = TOLERANCE_PROFILES[args.tolerance_profile]
    rtol = profile["rtol"] if args.compare_rtol is None else args.compare_rtol
    atol = profile["atol"] if args.compare_atol is None else args.compare_atol
    return float(rtol), float(atol)


def case_config(name: str, grad: np.ndarray, forw: Optional[np.ndarray]) -> Dict[str, Any]:
    if name == "norm":
        return {
            "processor_kwargs": {"norm_grad": True, "forw_illumination": False},
            "forw": None,
        }
    if name == "marine_smooth":
        return {
            "processor_kwargs": {
                "grad_mute": 2,
                "grad_smooth": 2,
                "grad_mask": None,
                "norm_grad": True,
                "forw_illumination": False,
                "marine_or_land": "marine",
            },
            "forw": None,
        }
    if name == "land_smooth":
        return {
            "processor_kwargs": {
                "grad_mute": 2,
                "grad_smooth": 2,
                "grad_mask": None,
                "norm_grad": True,
                "forw_illumination": False,
                "marine_or_land": "land",
            },
            "forw": None,
        }
    if name == "illumination":
        return {
            "processor_kwargs": {
                "grad_mute": 0,
                "grad_smooth": 0,
                "grad_mask": None,
                "norm_grad": True,
                "forw_illumination": True,
                "marine_or_land": "land",
            },
            "forw": forw,
        }
    if name == "mask_illumination":
        grad_mask = np.ones_like(grad, dtype=np.float32)
        grad_mask[: min(12, grad_mask.shape[0]), :] = 0.0
        return {
            "processor_kwargs": {
                "grad_mute": 0,
                "grad_smooth": 0,
                "grad_mask": grad_mask,
                "norm_grad": True,
                "forw_illumination": True,
                "marine_or_land": "land",
            },
            "forw": forw,
        }
    raise ValueError(f"unsupported case: {name}")


def make_inputs(args: argparse.Namespace, backend) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(args.seed)
    grad = rng.standard_normal((args.nz, args.nx)).astype(np.float32)
    grad += np.linspace(-0.5, 0.5, args.nz * args.nx, dtype=np.float32).reshape(args.nz, args.nx)
    forw = rng.uniform(0.2, 2.0, size=(args.nz, args.nx)).astype(np.float32)
    grad_torch = torch.as_tensor(grad, device=backend.device, dtype=backend.dtype)
    forw_torch = torch.as_tensor(forw, device=backend.device, dtype=backend.dtype)
    return grad, forw, grad_torch, forw_torch


def time_legacy(processor: GradProcessor, *, nx: int, nz: int, vmax: float, grad: np.ndarray, forw: Optional[np.ndarray]) -> Tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = processor.forward(nx=nx, nz=nz, vmax=vmax, grad=grad.copy(), forw=None if forw is None else forw.copy())
    return result, time.perf_counter() - start


def time_torch(
    processor: TorchGradProcessor,
    *,
    backend,
    nx: int,
    nz: int,
    vmax: float,
    grad: torch.Tensor,
    forw: Optional[torch.Tensor],
) -> Tuple[np.ndarray, float]:
    synchronize(backend)
    start = time.perf_counter()
    result = processor.forward_torch(
        nx=nx,
        nz=nz,
        vmax=torch.as_tensor(vmax, device=backend.device, dtype=backend.dtype),
        grad=grad,
        forw=forw,
    )
    synchronize(backend)
    elapsed = time.perf_counter() - start
    return result.detach().cpu().numpy(), elapsed


def run_case(
    args: argparse.Namespace,
    backend,
    name: str,
    grad_np,
    forw_np,
    grad_torch,
    forw_torch,
    *,
    compare_rtol: float,
    compare_atol: float,
) -> Dict[str, Any]:
    config = case_config(name, grad_np, forw_np)
    processor_kwargs = config["processor_kwargs"]
    case_forw_np = config["forw"]
    case_forw_torch = None if case_forw_np is None else forw_torch
    vmax = float(args.vmax)

    legacy_processor = GradProcessor(**processor_kwargs)
    torch_processor = TorchGradProcessor(**processor_kwargs)

    warmup_runs = []
    for _ in range(args.warmup):
        _, legacy_seconds = time_legacy(
            legacy_processor,
            nx=args.nx,
            nz=args.nz,
            vmax=vmax,
            grad=grad_np,
            forw=case_forw_np,
        )
        _, torch_seconds = time_torch(
            torch_processor,
            backend=backend,
            nx=args.nx,
            nz=args.nz,
            vmax=vmax,
            grad=grad_torch,
            forw=case_forw_torch,
        )
        warmup_runs.append({"legacy_seconds": legacy_seconds, "torch_seconds": torch_seconds})

    legacy_results = []
    torch_results = []
    runs = []
    for _ in range(args.repeat):
        legacy_result, legacy_seconds = time_legacy(
            legacy_processor,
            nx=args.nx,
            nz=args.nz,
            vmax=vmax,
            grad=grad_np,
            forw=case_forw_np,
        )
        torch_result, torch_seconds = time_torch(
            torch_processor,
            backend=backend,
            nx=args.nx,
            nz=args.nz,
            vmax=vmax,
            grad=grad_torch,
            forw=case_forw_torch,
        )
        legacy_results.append(legacy_result)
        torch_results.append(torch_result)
        runs.append({"legacy_seconds": legacy_seconds, "torch_seconds": torch_seconds})

    abs_diffs = [float(np.max(np.abs(torch_result - legacy_result))) for legacy_result, torch_result in zip(legacy_results, torch_results)]
    rel_diffs = [
        abs_diff / max(float(np.max(np.abs(legacy_result))), compare_atol)
        for abs_diff, legacy_result in zip(abs_diffs, legacy_results)
    ]
    max_abs_diff = max(abs_diffs)
    max_rel_diff = max(rel_diffs)
    failed = max_abs_diff > compare_atol and max_rel_diff > compare_rtol

    legacy_times = [run["legacy_seconds"] for run in runs]
    torch_times = [run["torch_seconds"] for run in runs]
    legacy_mean = summarize(legacy_times)["mean"]
    torch_mean = summarize(torch_times)["mean"]

    return {
        "case": name,
        "status": "failed" if failed else "ok",
        "processor_kwargs": serializable_processor_kwargs(processor_kwargs),
        "rtol": compare_rtol,
        "atol": compare_atol,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "legacy_seconds": summarize(legacy_times),
        "torch_seconds": summarize(torch_times),
        "speedup": None if torch_mean == 0 else legacy_mean / torch_mean,
        "runs": runs,
        "warmup_runs": warmup_runs if args.include_warmup else [],
    }


def serializable_processor_kwargs(processor_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    payload = {}
    for key, value in processor_kwargs.items():
        if isinstance(value, np.ndarray):
            payload[key] = {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "nonzero": int(np.count_nonzero(value)),
                "min": float(np.min(value)),
                "max": float(np.max(value)),
            }
        else:
            payload[key] = value
    return payload


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(requested_device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    grad_np, forw_np, grad_torch, forw_torch = make_inputs(args, backend)
    compare_rtol, compare_atol = resolve_tolerances(args)

    cases = [
        run_case(
            args,
            backend,
            name,
            grad_np,
            forw_np,
            grad_torch,
            forw_torch,
            compare_rtol=compare_rtol,
            compare_atol=compare_atol,
        )
        for name in args.cases
    ]
    status = "failed" if any(case["status"] != "ok" for case in cases) else "ok"
    return {
        "status": status,
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
            "cases": args.cases,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "nx": args.nx,
            "nz": args.nz,
            "vmax": args.vmax,
            "tolerance_profile": args.tolerance_profile,
            "compare_rtol": compare_rtol,
            "compare_atol": compare_atol,
        },
        "cases": cases,
    }


def write_report(report: Dict[str, Any], output: Optional[str]) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark legacy and torch-native gradient processors.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--cases", type=parse_case_list, default=parse_case_list("norm,marine_smooth,land_smooth,illumination"))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--nz", type=int, default=48)
    parser.add_argument("--vmax", type=float, default=2500.0)
    parser.add_argument(
        "--tolerance-profile",
        type=parse_tolerance_profile,
        default="strict",
        help="strict keeps CPU-level parity; npu-float32 accepts known NPU smoothing drift",
    )
    parser.add_argument("--compare-rtol", type=float, default=None, help="override the selected tolerance profile rtol")
    parser.add_argument("--compare-atol", type=float, default=None, help="override the selected tolerance profile atol")
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.nx <= 0 or args.nz <= 0:
        parser.error("--nx and --nz must be positive")
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
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
