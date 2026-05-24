#!/usr/bin/env python
"""Mini elastic inversion smoke test for bv1.2 backend behavior.

This script builds tiny in-memory true and initial isotropic elastic models,
synthesizes observed data with the true model, and runs one ElasticFWI iteration
on the initial model. It writes no notebooks, figures, wavefields, or example
outputs.
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

from ADFWI.backends import BackendUnavailableError, configure_backend
from ADFWI.fwi import ElasticFWI
from ADFWI.fwi.transforms import LegacyLowPassFilter, LowPassFilter
from ADFWI.fwi.misfit import Misfit_waveform_L2
from ADFWI.model import IsotropicElasticModel
from ADFWI.propagator import ElasticPropagator, GradProcessor
from ADFWI.survey import Receiver, SeismicData, Source, Survey


def parse_dtype(name: str) -> torch.dtype:
    dtypes = {"float32": torch.float32, "float64": torch.float64}
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
    source.add_source(nx // 2, max(3, nz // 4), ricker_wavelet(nt, dt, f0), src_type="mt")

    receiver = Receiver(nt=nt, dt=dt)
    rcv_x = np.array([nx // 4, nx // 2, (3 * nx) // 4], dtype=np.int64)
    rcv_z = np.full_like(rcv_x, max(3, nz // 4))
    receiver.add_receivers(rcv_x, rcv_z, rcv_type="pr")
    return Survey(source, receiver)


def model_arrays(nx: int, nz: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    nx: int,
    nz: int,
    dx: float,
    dz: float,
    nabc: int,
    vp_grad: bool,
) -> IsotropicElasticModel:
    return IsotropicElasticModel(
        0,
        0,
        nx,
        nz,
        dx,
        dz,
        vp,
        vs,
        rho,
        vp_grad=vp_grad,
        vs_grad=False,
        rho_grad=False,
        auto_update_rho=False,
        auto_update_vp=False,
        free_surface=False,
        nabc=nabc,
    )


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value.detach()).cpu().item())


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(requested_device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    true_vp, init_vp, true_vs, init_vs, rho = model_arrays(args.nx, args.nz)
    survey = build_survey(args.nt, args.dt, args.f0, args.nx, args.nz)

    true_model = build_model(true_vp, true_vs, rho, args.nx, args.nz, args.dx, args.dz, args.nabc, vp_grad=False)
    true_propagator = ElasticPropagator(true_model, survey)

    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        observed_record = true_propagator.forward(
            shot_index=np.array([0]), fd_order=args.fd_order, checkpoint_segments=args.checkpoint_segments
        )
    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    observed_seconds = time.perf_counter() - start

    obs_data = SeismicData(survey)
    obs_data.record_data(observed_record)

    model = build_model(init_vp, init_vs, rho, args.nx, args.nz, args.dx, args.dz, args.nabc, vp_grad=True)
    initial_vp = model.vp.detach().clone()
    propagator = ElasticPropagator(model, survey)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    gradient_processor = GradProcessor(norm_grad=False, forw_illumination=False)
    loss_fn = Misfit_waveform_L2(dt=args.dt)

    data_transform_pipeline = None
    legacy_cutoff_freq = None
    if args.lowpass_mode == "legacy":
        legacy_cutoff_freq = args.cutoff_freq
    elif args.lowpass_mode == "legacy-transform":
        data_transform_pipeline = LegacyLowPassFilter(cutoff_freq=args.cutoff_freq, dt=args.dt)
    elif args.lowpass_mode == "transform":
        data_transform_pipeline = LowPassFilter(cutoff_freq=args.cutoff_freq, dt=args.dt, filter_length=args.lowpass_filter_length)

    fwi = ElasticFWI(
        propagator,
        model,
        loss_fn,
        obs_data,
        optimizer=optimizer,
        scheduler=scheduler,
        gradient_processor=gradient_processor,
        waveform_normalize=False,
        data_transform_pipeline=data_transform_pipeline,
        cache_result=True,
        cache_result_epoch=1,
        save_fig_epoch=-1,
        inversion_component=["pressure"],
    )

    progress_context = nullcontext() if args.show_progress else redirect_stderr(io.StringIO())
    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    start = time.perf_counter()
    with progress_context:
        fwi.forward(iteration=1, batch_size=1, fd_order=args.fd_order, checkpoint_segments=args.checkpoint_segments, cutoff_freq=legacy_cutoff_freq)
    if backend.name in ("cuda", "npu"):
        backend.synchronize()
    inversion_seconds = time.perf_counter() - start

    if not fwi.iter_loss:
        raise RuntimeError("ElasticFWI did not record an iteration loss")
    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after one FWI iteration")

    final_vp = model.vp.detach()
    model_update_norm = tensor_norm(final_vp - initial_vp)
    grad_norm = tensor_norm(model.vp.grad)
    loss = float(fwi.iter_loss[-1])

    if not math.isfinite(loss):
        raise RuntimeError(f"loss is not finite: {loss}")
    if not math.isfinite(grad_norm) or grad_norm <= 0.0:
        raise RuntimeError(f"invalid vp gradient norm: {grad_norm}")
    if not math.isfinite(model_update_norm) or model_update_norm <= 0.0:
        raise RuntimeError(f"vp model did not update: {model_update_norm}")
    if model.vp.device != backend.device:
        raise RuntimeError(f"model device {model.vp.device} does not match backend {backend.device}")

    pressure_shape = list((-(observed_record["txx"] + observed_record["tzz"])).shape)
    return {
        "status": "ok",
        "backend": backend.diagnostics(),
        "model": {"nx": args.nx, "nz": args.nz, "nabc": args.nabc, "dx": args.dx, "dz": args.dz},
        "survey": {"shots": survey.source.num, "receivers": survey.receiver.num, "nt": args.nt, "dt": args.dt},
        "observed_forward": {
            "seconds": observed_seconds,
            "pressure_shape": pressure_shape,
            "fd_order": args.fd_order,
        },
        "inversion": {
            "iterations": 1,
            "loss": loss,
            "vp_grad_norm": grad_norm,
            "vp_update_norm": model_update_norm,
            "seconds": inversion_seconds,
            "lr": args.lr,
            "component": "pressure",
            "lowpass_mode": args.lowpass_mode,
            "cutoff_freq": args.cutoff_freq,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one tiny ElasticFWI iteration as a backend smoke test.")
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--cutoff-freq", type=float, default=None, help="optional low-pass cutoff frequency in Hz")
    parser.add_argument("--lowpass-mode", choices=("none", "legacy", "legacy-transform", "transform"), default="none", help="low-pass implementation to use when --cutoff-freq is set")
    parser.add_argument("--lowpass-filter-length", type=int, default=101, help="FIR length for --lowpass-mode transform")
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
    if args.lowpass_mode != "none" and args.cutoff_freq is None:
        parser.error("--cutoff-freq is required when --lowpass-mode is legacy or transform")
    if args.lowpass_mode == "none" and args.cutoff_freq is not None:
        parser.error("--lowpass-mode must be legacy or transform when --cutoff-freq is set")

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
