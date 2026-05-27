#!/usr/bin/env python
"""Stage-level diagnostics for legacy versus torch gradient processing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import BackendUnavailableError, configure_backend
from ADFWI.propagator import GradProcessor, TorchGradProcessor
from ADFWI.propagator.gradient_process import _torch_smooth2d, smooth2d
from scripts.benchmark.gradient_processor_benchmark import (
    environment_metadata,
    git_metadata,
    make_inputs,
    synchronize,
)
from scripts.smoke.acoustic_backend_smoke import parse_dtype


def array_stats(value: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(np.min(value)),
        "max": float(np.max(value)),
        "max_abs": float(np.max(np.abs(value))),
    }


def compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    diff = np.abs(candidate - reference)
    reference_scale = max(float(np.max(np.abs(reference))), 1e-12)
    return {
        "reference": array_stats(reference),
        "candidate": array_stats(candidate),
        "max_abs_diff": float(np.max(diff)),
        "max_rel_diff": float(np.max(diff) / reference_scale),
    }


def torch_to_numpy(value: torch.Tensor, backend) -> np.ndarray:
    synchronize(backend)
    return value.detach().cpu().numpy()


def torch_smooth_numpy(value: np.ndarray, span: int, *, device, dtype, backend) -> np.ndarray:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    result = _torch_smooth2d(tensor, span=span)
    return torch_to_numpy(result, backend)


def processor_result(
    processor: GradProcessor,
    *,
    nx: int,
    nz: int,
    vmax: float,
    grad: np.ndarray,
    forw: Optional[np.ndarray],
) -> np.ndarray:
    return processor.forward(nx=nx, nz=nz, vmax=vmax, grad=grad.copy(), forw=None if forw is None else forw.copy())


def torch_processor_result(
    processor: TorchGradProcessor,
    *,
    backend,
    nx: int,
    nz: int,
    vmax: float,
    grad: torch.Tensor,
    forw: Optional[torch.Tensor],
) -> np.ndarray:
    result = processor.forward_torch(
        nx=nx,
        nz=nz,
        vmax=torch.as_tensor(vmax, device=backend.device, dtype=backend.dtype),
        grad=grad,
        forw=forw,
    )
    return torch_to_numpy(result, backend)


def illumination_preconditioner_numpy(forw: np.ndarray, *, nx: int, nz: int) -> np.ndarray:
    span = 40 if min(nz, nx) > 40 else int(min(nz, nx) / 2)
    smoothed = smooth2d(forw, span)
    precond = smoothed / np.max(smoothed + 1e-5)
    precond[precond < 0.0001] = 0.0001
    return precond


def illumination_preconditioner_torch(forw: torch.Tensor, *, backend, nx: int, nz: int) -> np.ndarray:
    span = 40 if min(nz, nx) > 40 else int(min(nz, nx) / 2)
    smoothed = _torch_smooth2d(forw, span)
    epsilon = torch.as_tensor(0.0001, device=forw.device, dtype=forw.dtype)
    precond = smoothed / torch.max(smoothed + 1e-5)
    precond = torch.clamp(precond, min=epsilon)
    return torch_to_numpy(precond, backend)


def run_diagnostics(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(requested_device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    grad_np, forw_np, grad_torch, forw_torch = make_inputs(args, backend)

    cpu_backend = configure_backend("cpu", dtype=args.dtype, fallback=False)
    grad_torch_cpu = torch.as_tensor(grad_np, device=cpu_backend.device, dtype=cpu_backend.dtype)
    forw_torch_cpu = torch.as_tensor(forw_np, device=cpu_backend.device, dtype=cpu_backend.dtype)

    smooth_span = args.smooth_span
    legacy_smooth = smooth2d(grad_np, smooth_span).astype(np.float32)
    torch_cpu_smooth = torch_smooth_numpy(
        grad_np,
        smooth_span,
        device=cpu_backend.device,
        dtype=cpu_backend.dtype,
        backend=cpu_backend,
    )
    torch_device_smooth = torch_smooth_numpy(
        grad_np,
        smooth_span,
        device=backend.device,
        dtype=backend.dtype,
        backend=backend,
    )

    marine_no_norm_kwargs = {
        "grad_mute": 2,
        "grad_smooth": smooth_span,
        "norm_grad": False,
        "forw_illumination": False,
        "marine_or_land": "marine",
    }
    marine_norm_kwargs = dict(marine_no_norm_kwargs, norm_grad=True)
    illum_no_norm_kwargs = {
        "grad_mute": 0,
        "grad_smooth": 0,
        "norm_grad": False,
        "forw_illumination": True,
        "marine_or_land": "land",
    }
    illum_norm_kwargs = dict(illum_no_norm_kwargs, norm_grad=True)

    cases: Dict[str, Any] = {
        "smooth2d_raw": {
            "span": smooth_span,
            "legacy_vs_torch_cpu": compare_arrays(legacy_smooth, torch_cpu_smooth),
            "legacy_vs_torch_device": compare_arrays(legacy_smooth, torch_device_smooth),
            "torch_cpu_vs_torch_device": compare_arrays(torch_cpu_smooth, torch_device_smooth),
        },
        "marine_smooth_without_norm": compare_arrays(
            processor_result(
                GradProcessor(**marine_no_norm_kwargs),
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_np,
                forw=None,
            ),
            torch_processor_result(
                TorchGradProcessor(**marine_no_norm_kwargs),
                backend=backend,
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_torch,
                forw=None,
            ),
        ),
        "marine_smooth_with_norm": compare_arrays(
            processor_result(
                GradProcessor(**marine_norm_kwargs),
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_np,
                forw=None,
            ),
            torch_processor_result(
                TorchGradProcessor(**marine_norm_kwargs),
                backend=backend,
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_torch,
                forw=None,
            ),
        ),
        "illumination_preconditioner": compare_arrays(
            illumination_preconditioner_numpy(forw_np, nx=args.nx, nz=args.nz),
            illumination_preconditioner_torch(forw_torch, backend=backend, nx=args.nx, nz=args.nz),
        ),
        "illumination_without_norm": compare_arrays(
            processor_result(
                GradProcessor(**illum_no_norm_kwargs),
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_np,
                forw=forw_np,
            ),
            torch_processor_result(
                TorchGradProcessor(**illum_no_norm_kwargs),
                backend=backend,
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_torch,
                forw=forw_torch,
            ),
        ),
        "illumination_with_norm": compare_arrays(
            processor_result(
                GradProcessor(**illum_norm_kwargs),
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_np,
                forw=forw_np,
            ),
            torch_processor_result(
                TorchGradProcessor(**illum_norm_kwargs),
                backend=backend,
                nx=args.nx,
                nz=args.nz,
                vmax=args.vmax,
                grad=grad_torch,
                forw=forw_torch,
            ),
        ),
    }

    return {
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
            "nx": args.nx,
            "nz": args.nz,
            "vmax": args.vmax,
            "smooth_span": args.smooth_span,
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
    parser = argparse.ArgumentParser(description="Diagnose gradient processor drift by processing stage.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--vmax", type=float, default=2500.0)
    parser.add_argument("--smooth-span", type=int, default=2)
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.nx <= 0 or args.nz <= 0:
        parser.error("--nx and --nz must be positive")
    if args.smooth_span <= 0:
        parser.error("--smooth-span must be positive")
    try:
        report = run_diagnostics(args)
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
