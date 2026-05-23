#!/usr/bin/env python
"""Tensor-level misfit backend smoke tests for bv1.2.

This script creates small in-memory waveform tensors on a selected backend,
runs selected misfit functions, and checks finite loss plus finite nonzero
backward gradient on the synthetic waveform. It writes no files.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import BackendUnavailableError, configure_backend
from ADFWI.fwi.misfit import (
    Misfit_NIM,
    Misfit_global_correlation,
    Misfit_traveltime,
    Misfit_waveform_L1,
    Misfit_waveform_L2,
    Misfit_waveform_smoothL1,
    Misfit_waveform_studentT,
    Misfit_weighted_L1_and_L2,
)

try:
    from ADFWI.fwi.misfit import Misfit_envelope, Misfit_weighted_ECI
except Exception:  # pragma: no cover - optional smoke coverage only
    Misfit_envelope = None
    Misfit_weighted_ECI = None


MisfitFactory = Callable[[float], Any]


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {"float32": torch.float32, "float64": torch.float64}
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def parse_misfits(value: str) -> List[str]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        raise argparse.ArgumentTypeError("at least one misfit is required")
    return names


def build_waveforms(device: torch.device, dtype: torch.dtype, nshot: int, nt: int, nrec: int) -> Tuple[torch.Tensor, torch.Tensor]:
    t = torch.linspace(0.0, 1.0, nt, device=device, dtype=dtype)
    traces = []
    shifted = []
    for ishot in range(nshot):
        shot_traces = []
        shot_shifted = []
        for irec in range(nrec):
            freq = 2.0 + ishot + 0.5 * irec
            phase = 0.15 * (ishot + irec)
            obs_trace = torch.sin(2.0 * math.pi * freq * t + phase) * torch.exp(-1.5 * t)
            syn_trace = 0.92 * torch.sin(2.0 * math.pi * freq * (t - 0.015) + phase) * torch.exp(-1.3 * t)
            shot_traces.append(obs_trace)
            shot_shifted.append(syn_trace)
        traces.append(torch.stack(shot_traces, dim=-1))
        shifted.append(torch.stack(shot_shifted, dim=-1))
    obs = torch.stack(traces, dim=0)
    syn = torch.stack(shifted, dim=0).clone().detach().requires_grad_(True)
    return obs, syn


def make_registry(dt: float) -> Dict[str, Tuple[MisfitFactory, str]]:
    registry: Dict[str, Tuple[MisfitFactory, str]] = {
        "L1": (lambda dt: Misfit_waveform_L1(dt=dt), "forward"),
        "L2": (lambda dt: Misfit_waveform_L2(dt=dt), "forward"),
        "SmoothL1": (lambda dt: Misfit_waveform_smoothL1(dt=dt), "forward"),
        "StudentT": (lambda dt: Misfit_waveform_studentT(dt=dt), "forward"),
        "WeightedL1L2": (lambda dt: Misfit_weighted_L1_and_L2(dt=dt, max_iter=4), "forward"),
        "GC": (lambda dt: Misfit_global_correlation(dt=dt), "forward"),
        "TravelTime": (lambda dt: Misfit_traveltime(dt=dt, beta=5), "forward"),
        "NIM": (lambda dt: Misfit_NIM(p=1, trans_type="linear", theta=1, dt=dt), "nim_apply"),
    }
    if Misfit_envelope is not None:
        registry["Envelope"] = (lambda dt: Misfit_envelope(dt=dt, p=1.5, instaneous_phase=False), "forward")
    if Misfit_weighted_ECI is not None:
        registry["WECI"] = (lambda dt: Misfit_weighted_ECI(max_iter=4, dt=dt, p=1.5, instaneous_phase=False), "forward")
    return registry


def call_misfit(fn: Any, mode: str, obs: torch.Tensor, syn: torch.Tensor) -> torch.Tensor:
    if mode == "nim_apply":
        return fn.apply(syn, obs, fn.p, fn.trans_type, fn.theta)
    return fn.forward(obs, syn)


def run_one(name: str, factory: MisfitFactory, mode: str, args: argparse.Namespace, backend_device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
    obs, syn = build_waveforms(backend_device, dtype, args.nshot, args.nt, args.nrec)
    fn = factory(args.dt)
    start = time.perf_counter()
    loss = call_misfit(fn, mode, obs, syn)
    if loss.ndim != 0:
        loss = loss.sum()
    loss.backward()
    elapsed = time.perf_counter() - start

    grad = syn.grad
    if grad is None:
        raise RuntimeError("synthetic gradient is None")
    loss_value = float(loss.detach().cpu().item())
    grad_norm = float(torch.linalg.norm(grad.detach()).cpu().item())
    loss_isfinite = bool(torch.isfinite(loss.detach()).cpu().item())
    grad_isfinite = bool(torch.isfinite(grad.detach()).all().cpu().item())
    grad_nonzero = bool(grad_norm > 0.0)
    if not loss_isfinite:
        raise RuntimeError(f"loss is not finite: {loss_value}")
    if not grad_isfinite:
        raise RuntimeError("gradient contains NaN or Inf")
    if not grad_nonzero:
        raise RuntimeError("gradient norm is zero")
    return {
        "status": "ok",
        "loss": loss_value,
        "grad_norm": grad_norm,
        "loss_isfinite": loss_isfinite,
        "grad_isfinite": grad_isfinite,
        "grad_nonzero": grad_nonzero,
        "seconds": elapsed,
    }


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(requested_device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)
    torch.manual_seed(args.seed)

    registry = make_registry(args.dt)
    runs: Dict[str, Any] = {}
    for name in args.misfits:
        if name not in registry:
            runs[name] = {"status": "unknown", "error": f"unknown or unavailable misfit: {name}"}
            continue
        factory, mode = registry[name]
        try:
            runs[name] = run_one(name, factory, mode, args, backend.device, backend.dtype)
        except Exception as exc:
            runs[name] = {"status": "failed", "error": repr(exc)}

    failed = [name for name, result in runs.items() if result["status"] not in ("ok",)]
    return {
        "status": "failed" if failed else "ok",
        "backend": backend.diagnostics(),
        "shape": {"nshot": args.nshot, "nt": args.nt, "nrec": args.nrec, "dt": args.dt},
        "misfits": args.misfits,
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tensor-level misfit backend smoke tests.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--misfits", type=parse_misfits, default=parse_misfits("L1,L2,SmoothL1,StudentT,WeightedL1L2,GC,TravelTime,NIM"))
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nshot", type=int, default=2)
    parser.add_argument("--nt", type=int, default=32)
    parser.add_argument("--nrec", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.001)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_smoke(args)
    except BackendUnavailableError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
