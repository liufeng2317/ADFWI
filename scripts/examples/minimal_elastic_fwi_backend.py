#!/usr/bin/env python
"""Minimal elastic FWI example using the bv1.2 backend API.

This script mirrors the minimal acoustic backend example for isotropic elastic
FWI. It configures the backend once, builds tiny true/initial elastic models,
synthesizes observed data in memory, runs one ElasticFWI iteration, and prints a
JSON summary. It writes no files, figures, notebooks, or wavefields.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from contextlib import nullcontext, redirect_stderr
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import ADFWI
from ADFWI.backends import BackendUnavailableError
from ADFWI.fwi import ElasticFWI
from ADFWI.fwi.misfit import Misfit_waveform_L2
from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator, GradProcessor
from ADFWI.survey import Receiver, SeismicData, Source, Survey


def ricker_wavelet(nt: int, dt: float, f0: float) -> np.ndarray:
    """Return a Ricker wavelet sampled for ADFWI source input."""
    t = np.arange(nt, dtype=np.float32) * dt
    t0 = 1.0 / f0
    arg = math.pi * f0 * (t - t0)
    return ((1.0 - 2.0 * arg * arg) * np.exp(-(arg * arg))).astype(np.float32)


def build_survey(nt: int, dt: float, f0: float, nx: int, nz: int) -> Survey:
    """Create one source and three pressure receivers on a tiny grid."""
    source = Source(nt=nt, dt=dt, f0=f0)
    source.add_source(nx // 2, max(3, nz // 4), ricker_wavelet(nt, dt, f0), src_type="mt")

    receiver = Receiver(nt=nt, dt=dt)
    rcv_x = np.array([nx // 4, nx // 2, (3 * nx) // 4], dtype=np.int64)
    rcv_z = np.full_like(rcv_x, max(3, nz // 4))
    receiver.add_receivers(rcv_x, rcv_z, rcv_type="pr")
    return Survey(source, receiver)


def model_arrays(nx: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build small true/initial elastic fields and a fixed rho field."""
    x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
    z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
    xx, zz = np.meshgrid(x, z)
    anomaly = np.exp(-8.0 * (xx * xx + (zz - 0.55) ** 2)).astype(np.float32)
    true_vp = 2200.0 + 160.0 * zz + 60.0 * anomaly
    init_vp = 2200.0 + 160.0 * zz + 20.0 * anomaly
    true_vs = 1200.0 + 80.0 * zz + 30.0 * anomaly
    init_vs = true_vs.copy()
    rho = np.full((nz, nx), 2000.0, dtype=np.float32)
    return true_vp.astype(np.float32), init_vp.astype(np.float32), true_vs.astype(np.float32), init_vs.astype(np.float32), rho


def build_model(
    vp: np.ndarray,
    vs: np.ndarray,
    rho: np.ndarray,
    args: argparse.Namespace,
    *,
    vp_grad: bool,
) -> IsotropicElasticModel:
    """Create an IsotropicElasticModel that inherits the active ADFWI backend."""
    return IsotropicElasticModel(
        0,
        0,
        args.nx,
        args.nz,
        args.dx,
        args.dz,
        vp,
        vs,
        rho,
        vp_grad=vp_grad,
        vs_grad=False,
        rho_grad=False,
        auto_update_rho=False,
        auto_update_vp=False,
        free_surface=False,
        nabc=args.nabc,
    )


def tensor_norm(value: torch.Tensor) -> float:
    """Return a finite Python float norm for JSON summaries."""
    return float(torch.linalg.norm(value.detach()).cpu().item())


def configure_example_backend(args: argparse.Namespace):
    """Configure the recommended top-level backend API for this example."""
    device: Optional[str] = None if args.device == "auto" else args.device
    prefer = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    return ADFWI.set_backend(device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)


def elastic_pressure(record: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return the pressure component used by ElasticFWI."""
    return -(record["txx"] + record["tzz"])


def run_example(args: argparse.Namespace) -> Dict[str, Any]:
    """Run one tiny in-memory ElasticFWI iteration and return a JSON-ready summary."""
    backend = configure_example_backend(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    true_vp, init_vp, true_vs, init_vs, rho = model_arrays(args.nx, args.nz)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)

    true_model = build_model(true_vp, true_vs, rho, args, vp_grad=False)
    true_propagator = ElasticPropagator(true_model, survey)

    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        observed_record = true_propagator.forward(
            shot_index=np.array([0]),
            fd_order=args.fd_order,
            checkpoint_segments=args.checkpoint_segments,
        )
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    observed_seconds = time.perf_counter() - start

    obs_data = SeismicData(survey)
    obs_data.record_data(observed_record)

    model = build_model(init_vp, init_vs, rho, args, vp_grad=True)
    initial_vp = model.vp.detach().clone()
    propagator = ElasticPropagator(model, survey)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    loss_fn = Misfit_waveform_L2(dt=args.dt)
    gradient_processor = GradProcessor(norm_grad=False, forw_illumination=False)

    fwi = ElasticFWI(
        propagator,
        model,
        loss_fn,
        obs_data,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_processor=gradient_processor,
        waveform_normalize=False,
        cache_result=True,
        cache_result_epoch=1,
        save_fig_epoch=-1,
        inversion_component=["pressure"],
    )

    progress_context = nullcontext() if args.show_progress else redirect_stderr(io.StringIO())
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with progress_context:
        fwi.forward(
            iteration=1,
            batch_size=1,
            fd_order=args.fd_order,
            checkpoint_segments=args.checkpoint_segments,
        )
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    inversion_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after one ElasticFWI iteration")
    if not fwi.iter_loss:
        raise RuntimeError("ElasticFWI did not record an iteration loss")

    loss = float(fwi.iter_loss[-1])
    vp_grad_norm = tensor_norm(model.vp.grad)
    vp_update_norm = tensor_norm(model.vp.detach() - initial_vp)
    if not math.isfinite(loss):
        raise RuntimeError(f"loss is not finite: {loss}")
    if not math.isfinite(vp_grad_norm) or vp_grad_norm <= 0.0:
        raise RuntimeError(f"invalid vp gradient norm: {vp_grad_norm}")
    if not math.isfinite(vp_update_norm) or vp_update_norm <= 0.0:
        raise RuntimeError(f"model update norm is invalid: {vp_update_norm}")

    pressure = elastic_pressure(observed_record)
    return {
        "status": "ok",
        "backend": ADFWI.backend_diagnostics(),
        "model": {
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "dx": args.dx,
            "dz": args.dz,
            "vp_device": str(model.vp.device),
            "vp_dtype": str(model.vp.dtype).replace("torch.", ""),
        },
        "survey": {
            "shots": survey.source.num,
            "receivers": survey.receiver.num,
            "nt": args.nt,
            "dt": args.dt,
        },
        "observed_forward": {
            "seconds": observed_seconds,
            "pressure_shape": list(pressure.shape),
            "fd_order": args.fd_order,
        },
        "inversion": {
            "iterations": 1,
            "components": ["pressure"],
            "misfit": "L2",
            "optimizer": "SGD",
            "lr": args.lr,
            "seconds": inversion_seconds,
            "loss": loss,
            "vp_grad_norm": vp_grad_norm,
            "vp_update_norm": vp_update_norm,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority used when --device auto")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default="float32", help="backend dtype string accepted by ADFWI.set_backend")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--fd-order", type=int, default=4, choices=(4, 6, 8, 10))
    parser.add_argument("--show-progress", action="store_true", help="show ElasticFWI tqdm progress bars")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--lr", type=float, default=1e5)
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
        result = run_example(args)
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
