#!/usr/bin/env python
"""Acoustic backend smoke test for bv1.2 device behavior.

This script runs a tiny in-memory acoustic forward pass and, by default, a
single scalar backward pass. It intentionally does not read/write notebooks,
figures, wavefields, or example outputs.

Examples
--------
python scripts/smoke/acoustic_backend_smoke.py --device cpu
python scripts/smoke/acoustic_backend_smoke.py --device npu:0
python scripts/smoke/acoustic_backend_smoke.py --device auto --prefer npu,cpu
"""

from __future__ import annotations

import argparse
import json
import math
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
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator
from ADFWI.survey import Receiver, Source, Survey


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unsupported dtype: {name}") from exc


def ricker_wavelet(nt: int, dt: float, f0: float) -> np.ndarray:
    t = np.arange(nt, dtype=np.float32) * dt
    t0 = 1.0 / f0
    arg = math.pi * f0 * (t - t0)
    return ((1.0 - 2.0 * arg * arg) * np.exp(-(arg * arg))).astype(np.float32)


def build_survey(nt: int, dt: float, f0: float, nx: int, nz: int) -> Survey:
    source = Source(nt=nt, dt=dt, f0=f0)
    source.add_source(nx // 2, max(2, nz // 4), ricker_wavelet(nt, dt, f0), src_type="mt")

    receiver = Receiver(nt=nt, dt=dt)
    rcv_x = np.array([nx // 4, nx // 2, (3 * nx) // 4], dtype=np.int64)
    rcv_z = np.full_like(rcv_x, max(2, nz // 4))
    receiver.add_receivers(rcv_x, rcv_z, rcv_type="pr")
    return Survey(source, receiver)


def build_model(nx: int, nz: int, dx: float, dz: float, nabc: int) -> AcousticModel:
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    vp = 1800.0 + 120.0 * zz + 40.0 * np.exp(-8.0 * (xx * xx + (zz - 0.55) ** 2))
    rho = np.full((nz, nx), 1900.0, dtype=np.float32)
    return AcousticModel(
        0,
        0,
        nx,
        nz,
        dx,
        dz,
        vp.astype(np.float32),
        rho,
        vp_grad=True,
        rho_grad=False,
        auto_update_rho=False,
        nabc=nabc,
    )


def scalar_float(value: torch.Tensor) -> float:
    return float(value.detach().cpu().item())


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
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

    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)
    propagator = AcousticPropagator(model, survey)

    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    start = time.perf_counter()
    record = propagator.forward(shot_index=np.array([0]), checkpoint_segments=args.checkpoint_segments)
    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    forward_seconds = time.perf_counter() - start

    pressure = record["p"]
    if pressure.shape != (1, args.nt, 3):
        raise RuntimeError(f"unexpected pressure shape: {tuple(pressure.shape)}")
    if pressure.device != backend.device:
        raise RuntimeError(f"pressure device {pressure.device} does not match backend {backend.device}")
    if pressure.dtype != backend.dtype:
        raise RuntimeError(f"pressure dtype {pressure.dtype} does not match backend {backend.dtype}")
    if not torch.isfinite(pressure).all():
        raise RuntimeError("forward pressure contains NaN or Inf")

    loss = pressure.pow(2).mean()
    grad_norm = None
    backward_seconds = None
    grad_isfinite = None
    grad_nonzero = None

    if not args.skip_backward:
        if backend.name in ("cuda", "npu"):
            backend.synchronize()
        start = time.perf_counter()
        loss.backward()
        if backend.name in ("cuda", "npu"):
            backend.synchronize()
        backward_seconds = time.perf_counter() - start

        if model.vp.grad is None:
            raise RuntimeError("model.vp.grad is None after backward")
        grad = model.vp.grad.detach()
        grad_isfinite = bool(torch.isfinite(grad).all().detach().cpu().item())
        grad_norm = float(torch.linalg.norm(grad).detach().cpu().item())
        grad_nonzero = bool(grad_norm > 0.0)
        if not grad_isfinite:
            raise RuntimeError("vp gradient contains NaN or Inf")
        if not grad_nonzero:
            raise RuntimeError("vp gradient norm is zero")

    return {
        "status": "ok",
        "backend": backend.diagnostics(),
        "model": {
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "dx": args.dx,
            "dz": args.dz,
        },
        "survey": {
            "shots": survey.source.num,
            "receivers": survey.receiver.num,
            "nt": args.nt,
            "dt": args.dt,
        },
        "forward": {
            "pressure_shape": list(pressure.shape),
            "pressure_dtype": str(pressure.dtype).replace("torch.", ""),
            "pressure_device": str(pressure.device),
            "pressure_l2": float(torch.linalg.norm(pressure).detach().cpu().item()),
            "loss": scalar_float(loss),
            "seconds": forward_seconds,
        },
        "backward": {
            "enabled": not args.skip_backward,
            "vp_grad_norm": grad_norm,
            "vp_grad_isfinite": grad_isfinite,
            "vp_grad_nonzero": grad_nonzero,
            "seconds": backward_seconds,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a tiny acoustic forward/backward backend smoke test.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--skip-backward", action="store_true", help="only run forward")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
