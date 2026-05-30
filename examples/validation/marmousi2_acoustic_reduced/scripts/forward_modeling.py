#!/usr/bin/env python
"""Notebook-equivalent Marmousi2 acoustic forward-modeling validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = VALIDATION_ROOT / "outputs" / "minimal_notebook"


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default) + "\n")


def import_runtime_modules():
    import numpy as np
    import torch

    from scipy import integrate

    os.chdir(REPO_ROOT)

    import ADFWI
    from ADFWI.model import AcousticModel
    from ADFWI.propagator import AcousticPropagator
    from ADFWI.survey import Receiver, SeismicData, Source, Survey
    from ADFWI.utils import load_marmousi_model, resample_marmousi_model, wavelet
    from ADFWI.view import plot_damp

    return {
        "np": np,
        "torch": torch,
        "integrate": integrate,
        "ADFWI": ADFWI,
        "AcousticModel": AcousticModel,
        "AcousticPropagator": AcousticPropagator,
        "Receiver": Receiver,
        "SeismicData": SeismicData,
        "Source": Source,
        "Survey": Survey,
        "load_marmousi_model": load_marmousi_model,
        "resample_marmousi_model": resample_marmousi_model,
        "wavelet": wavelet,
        "plot_damp": plot_damp,
    }


def add_case_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="npu:0", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-dir", type=Path, default=REPO_ROOT / "examples" / "datasets" / "marmousi2_source")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--ox", type=float, default=0.0)
    parser.add_argument("--oz", type=float, default=0.0)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--dx", type=float, default=40.0)
    parser.add_argument("--dz", type=float, default=40.0)
    parser.add_argument("--nt", type=int, default=3000)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--nabc", type=int, default=30)
    parser.add_argument("--f0", type=float, default=5.0)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--abc-type", default="PML")
    parser.add_argument("--abc-jerjan-alpha", type=float, default=0.007)


def validate_case_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.shots <= 0:
        parser.error("--shots must be positive")


def ensure_forward_dirs(output_root: Path) -> None:
    for subdir in ("model", "waveform", "survey"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def cumulative_source_wavelet(rt: Dict[str, Any], nt: int, dt: float, f0: float):
    integrate = rt["integrate"]
    _, src_v = rt["wavelet"](nt, dt, f0, amp0=1)
    try:
        return integrate.cumtrapz(src_v, axis=-1, initial=0)
    except AttributeError:
        return integrate.cumulative_trapezoid(src_v, axis=-1, initial=0)


def build_survey(rt: Dict[str, Any], args: argparse.Namespace):
    np = rt["np"]
    source = rt["Source"](nt=args.nt, dt=args.dt, f0=args.f0)
    src_z = np.array([1 for _ in range(2, args.nx - 1, 5)])[: args.shots]
    src_x = np.array([i for i in range(2, args.nx - 1, 5)])[: args.shots]
    src_v = cumulative_source_wavelet(rt, args.nt, args.dt, args.f0)
    for i in range(len(src_x)):
        source.add_source(
            src_x=src_x[i],
            src_z=src_z[i],
            src_wavelet=src_v,
            src_type="mt",
            src_mt=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        )

    receiver = rt["Receiver"](nt=args.nt, dt=args.dt)
    rcv_z = np.array([1 for _ in range(0, args.nx, 1)])
    rcv_x = np.array([j for j in range(0, args.nx, 1)])
    for i in range(len(rcv_x)):
        receiver.add_receiver(rcv_x=rcv_x[i], rcv_z=rcv_z[i], rcv_type="pr")

    return rt["Survey"](source=source, receiver=receiver)


def build_true_arrays(rt: Dict[str, Any], args: argparse.Namespace):
    np = rt["np"]
    marmousi_model = rt["load_marmousi_model"](in_dir=str(args.dataset_dir))
    x = np.linspace(5000, 5000 + args.dx * args.nx, args.nx)
    z = np.linspace(0, args.dz * args.nz, args.nz)
    true_model = rt["resample_marmousi_model"](x, z, marmousi_model)
    vp_true = true_model["vp"].T
    rho_true = np.power(vp_true, 0.25) * 310
    return true_model, vp_true, rho_true


def build_true_model(rt: Dict[str, Any], args: argparse.Namespace):
    _, vp_true, rho_true = build_true_arrays(rt, args)
    return rt["AcousticModel"](
        args.ox,
        args.oz,
        args.nx,
        args.nz,
        args.dx,
        args.dz,
        vp_true,
        rho_true,
        vp_grad=False,
        free_surface=args.free_surface,
        abc_type=args.abc_type,
        abc_jerjan_alpha=args.abc_jerjan_alpha,
        nabc=args.nabc,
        device=args.device,
        dtype=rt["torch"].float32 if args.dtype == "float32" else rt["torch"].float64,
    )


def tensor_summary(rt: Dict[str, Any], value) -> Dict[str, Any]:
    torch = rt["torch"]
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "device": str(detached.device),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(detached).all().item()),
        "min": float(detached.min().cpu().item()),
        "max": float(detached.max().cpu().item()),
        "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
    }


def array_summary(rt: Dict[str, Any], value) -> Dict[str, Any]:
    np = rt["np"]
    array = np.asarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "norm": float(np.linalg.norm(array.reshape(-1))),
    }


def save_forward_figures(rt: Dict[str, Any], output_root: Path, model, survey, propagator, d_obs) -> None:
    survey.source.plot_wavelet(save_path=str(output_root / "survey" / "wavelets.png"))
    survey.plot(model.vp, cmap="coolwarm", save_path=str(output_root / "survey" / "observed_system.png"))
    rt["plot_damp"](propagator.damp, save_path=str(output_root / "model" / "boundary_condition.png"))
    d_obs.plot_waveform2D(
        i_shot=0,
        rcv_type="pressure",
        acoustic_or_elastic="acoustic",
        normalize=False,
        figsize=(6, 6),
        cmap="coolwarm",
        save_path=str(output_root / "waveform" / "obs_2D_shot_0.png"),
        show=False,
    )
    d_obs.plot_waveform_wiggle(
        i_shot=0,
        rcv_type="pressure",
        acoustic_or_elastic="acoustic",
        normalize=False,
        save_path=str(output_root / "waveform" / "obs_wiggle_shot_0.png"),
        show=False,
    )
    d_obs.plot_waveform_trace(
        i_shot=0,
        i_trace=min(100, survey.receiver.num - 1),
        rcv_type="pressure",
        acoustic_or_elastic="acoustic",
        normalize=False,
        save_path=str(output_root / "waveform" / "obs_trace_shot_0_trace_100.png"),
        show=False,
    )


def run_check(args: argparse.Namespace) -> Dict[str, Any]:
    rt = import_runtime_modules()
    rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    ensure_forward_dirs(args.output_root)
    model = build_true_model(rt, args)
    survey = build_survey(rt, args)
    propagator = rt["AcousticPropagator"](model, survey, device=args.device)
    return {
        "status": "ok",
        "stage": "check",
        "output_root": str(args.output_root),
        "backend": rt["ADFWI"].backend_diagnostics(),
        "model": {"vp": tensor_summary(rt, model.vp), "rho": tensor_summary(rt, model.rho)},
        "survey": {"shots": survey.source.num, "receivers": survey.receiver.num, "nt": survey.source.nt, "dt": survey.source.dt},
        "propagator": {"device": str(propagator.device), "dtype": str(propagator.dtype).replace("torch.", "")},
    }


def run_forward(args: argparse.Namespace) -> Dict[str, Any]:
    rt = import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    torch = rt["torch"]
    ensure_forward_dirs(args.output_root)

    model = build_true_model(rt, args)
    model.save(str(args.output_root / "model" / "true_model.npz"))
    model._plot_vp_rho(
        figsize=(12, 5),
        wspace=0.15,
        cbar_pad_fraction=0.02,
        cmap="coolwarm",
        save_path=str(args.output_root / "model" / "true_vp_rho.png"),
    )

    survey = build_survey(rt, args)
    propagator = rt["AcousticPropagator"](model, survey, device=args.device)
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        record_waveform = propagator.forward(checkpoint_segments=args.checkpoint_segments)
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    seconds = time.perf_counter() - start

    d_obs = rt["SeismicData"](survey)
    d_obs.record_data(record_waveform)
    obs_path = args.output_root / "waveform" / "obs_data.npz"
    d_obs.save(str(obs_path))
    save_forward_figures(rt, args.output_root, model, survey, propagator, d_obs)

    report = {
        "status": "ok",
        "stage": "forward",
        "output_root": str(args.output_root),
        "obs_data": str(obs_path),
        "backend": rt["ADFWI"].backend_diagnostics(),
        "seconds": seconds,
        "checkpoint_segments": args.checkpoint_segments,
        "survey": {"shots": survey.source.num, "receivers": survey.receiver.num, "nt": survey.source.nt, "dt": survey.source.dt},
        "record": {"p": array_summary(rt, d_obs.data["p"])},
    }
    write_json(args.output_root / "forward_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "forward"), nargs="?", default="forward")
    add_case_arguments(parser)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_case_args(parser, args)
    try:
        report = run_check(args) if args.stage == "check" else run_forward(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "stage": args.stage, "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
