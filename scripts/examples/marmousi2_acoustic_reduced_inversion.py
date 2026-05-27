#!/usr/bin/env python
"""Run a reduced Marmousi2 acoustic inversion smoke with the bv1.2 backend API.

This script is a small inversion check for the existing Marmousi2
acoustic case. It reads the saved model and observed waveform files, selects a
small shot/time subset, runs a configurable number of AcousticFWI iterations,
and prints a JSON summary. When `--output-dir` is set, it also writes compact
JSON, CSV, and PNG artifacts for full-case comparison.
"""

from __future__ import annotations

import argparse
import csv
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import torch

import ADFWI
from ADFWI.backends import BackendUnavailableError
from ADFWI.fwi import AcousticFWI
from ADFWI.fwi.misfit import Misfit_waveform_L2, Misfit_waveform_SquaredL2
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator, GradProcessor, TorchGradProcessor
from ADFWI.survey import SeismicData

from marmousi2_acoustic_backend_check import (
    DEFAULT_CASE_DIR,
    build_survey,
    configure_backend,
    load_npz,
    optional_bound,
    parse_dtype,
    tensor_summary,
)



def build_loss(args: argparse.Namespace):
    if args.misfit == "safe-squared-l2":
        return Misfit_waveform_SquaredL2(dt=args.dt_for_loss, reduction="mean"), "safe-squared-l2"
    if args.misfit == "legacy-l2":
        return Misfit_waveform_L2(dt=args.dt_for_loss), "legacy-l2"
    raise ValueError(f"unsupported misfit: {args.misfit}")


def build_optimizer(model, args: argparse.Namespace):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr), "SGD"
    if args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr), "Adam"
    raise ValueError(f"unsupported optimizer: {args.optimizer}")


def build_gradient_processor(args: argparse.Namespace, grad_mask: np.ndarray):
    processor_kwargs = {
        "grad_mute": args.grad_mute,
        "grad_smooth": args.grad_smooth,
        "grad_mask": grad_mask,
        "norm_grad": args.norm_grad,
        "forw_illumination": args.forw_illumination,
        "marine_or_land": args.marine_or_land,
    }
    if args.gradient_processor == "legacy":
        return GradProcessor(**processor_kwargs)
    if args.gradient_processor == "torch":
        return TorchGradProcessor(**processor_kwargs)
    raise ValueError(f"unsupported gradient processor: {args.gradient_processor}")


def build_case_model(model_npz: Any, args: argparse.Namespace, *, vp_grad: bool, auto_update_rho: bool) -> AcousticModel:
    vp = np.asarray(model_npz["vp"], dtype=np.float32)
    rho = np.asarray(model_npz["rho"], dtype=np.float32)
    return AcousticModel(
        float(model_npz["ox"]),
        float(model_npz["oz"]),
        int(model_npz["nx"]),
        int(model_npz["nz"]),
        float(model_npz["dx"]),
        float(model_npz["dz"]),
        vp,
        rho,
        vp_bound=optional_bound(model_npz["vp_bound"]),
        rho_bound=optional_bound(model_npz["rho_bound"]),
        vp_grad=vp_grad,
        rho_grad=False,
        auto_update_rho=auto_update_rho,
        free_surface=bool(model_npz["free_surface"]),
        abc_type=args.abc_type,
        abc_jerjan_alpha=args.abc_jerjan_alpha,
        nabc=int(model_npz["nabc"]),
    )


def build_inversion_model(model_npz: Any, args: argparse.Namespace) -> AcousticModel:
    return build_case_model(model_npz, args, vp_grad=True, auto_update_rho=args.auto_update_rho)


def subset_obs_npz(obs_npz: Any, *, shot_count: int, nt_samples: int) -> Dict[str, Any]:
    full_nt = int(obs_npz["nt"])
    full_src_num = int(obs_npz["src_num"])
    if shot_count <= 0 or shot_count > full_src_num:
        raise ValueError(f"shot_count must be in [1, {full_src_num}], got {shot_count}")
    if nt_samples <= 0 or nt_samples > full_nt:
        raise ValueError(f"nt_samples must be in [1, {full_nt}], got {nt_samples}")

    dt = float(obs_npz["dt"])
    return {
        "src_loc": np.asarray(obs_npz["src_loc"], dtype=np.int64)[:shot_count].copy(),
        "rcv_loc": np.asarray(obs_npz["rcv_loc"], dtype=np.int64).copy(),
        "src_type": np.asarray(obs_npz["src_type"]).astype(str)[:shot_count].copy(),
        "rcv_type": np.asarray(obs_npz["rcv_type"]).astype(str).copy(),
        "nt": int(nt_samples),
        "dt": dt,
        "t": np.arange(nt_samples, dtype=np.float64) * dt,
    }


def build_observed_data(survey, obs_npz: Any, *, shot_count: int, nt_samples: int) -> SeismicData:
    raw = obs_npz["data"].item()
    obs_data = SeismicData(survey)
    obs_data.data = {}
    for key in ("p", "u", "w"):
        if key in raw:
            obs_data.data[key] = np.asarray(raw[key])[:shot_count, :nt_samples, :].copy()
    if "p" not in obs_data.data:
        raise RuntimeError("Marmousi2 observed data does not contain pressure key 'p'")
    return obs_data


def synthesize_observed_data(true_model_npz: Any, survey, args: argparse.Namespace, backend) -> tuple[SeismicData, float]:
    true_model = build_case_model(true_model_npz, args, vp_grad=False, auto_update_rho=False)
    true_propagator = AcousticPropagator(true_model, survey)
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        observed_record = true_propagator.forward(
            shot_index=np.arange(args.shot_count),
            checkpoint_segments=args.checkpoint_segments,
        )
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    seconds = time.perf_counter() - start
    obs_data = SeismicData(survey)
    obs_data.record_data(observed_record)
    return obs_data, seconds


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.linalg.norm(value.detach().reshape(-1)).cpu().item())


def finite_positive(value: float, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label} must be finite and positive, got {value}")


def finite_value(value: float, label: str) -> None:
    if not math.isfinite(value):
        raise RuntimeError(f"{label} must be finite, got {value}")


def summarize_losses(losses):
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_delta = final_loss - initial_loss
    return {
        "initial_loss": initial_loss,
        "loss": final_loss,
        "loss_history": losses,
        "loss_min": min(losses),
        "loss_max": max(losses),
        "loss_delta": loss_delta,
        "loss_relative_delta": loss_delta / max(abs(initial_loss), 1e-30),
    }


def write_outputs(output_dir: Path, report: Dict[str, Any], initial_vp: torch.Tensor, final_vp: torch.Tensor, losses) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    loss_csv_path = output_dir / "loss_history.csv"
    loss_png_path = output_dir / "loss_curve.png"
    vp_png_path = output_dir / "vp_initial_final_delta.png"

    summary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    with loss_csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "loss"])
        for idx, value in enumerate(losses):
            writer.writerow([idx, value])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(losses)), losses, color="black", linewidth=1.8)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Marmousi2 Reduced Inversion Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_png_path, dpi=150)
    plt.close()

    initial = initial_vp.detach().cpu().numpy()
    final = final_vp.detach().cpu().numpy()
    delta = final - initial
    vmin = float(min(initial.min(), final.min()))
    vmax = float(max(initial.max(), final.max()))
    dmax = float(np.max(np.abs(delta)))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(initial, cmap="jet_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Initial vp")
    im1 = axes[1].imshow(final, cmap="jet_r", vmin=vmin, vmax=vmax)
    axes[1].set_title("Final vp")
    im2 = axes[2].imshow(delta, cmap="coolwarm", vmin=-dmax, vmax=dmax)
    axes[2].set_title("Final - initial")
    for ax in axes:
        ax.set_xlabel("x index")
        ax.set_ylabel("z index")
    fig.colorbar(im0, ax=axes[:2], shrink=0.82, label="vp")
    fig.colorbar(im2, ax=axes[2], shrink=0.82, label="delta vp")
    fig.savefig(vp_png_path, dpi=150)
    plt.close(fig)

    return {
        "summary_json": str(summary_path),
        "loss_history_csv": str(loss_csv_path),
        "loss_curve_png": str(loss_png_path),
        "vp_initial_final_delta_png": str(vp_png_path),
    }


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    backend = configure_backend(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    case_dir = args.case_dir.resolve()
    model_npz = load_npz(case_dir / "data" / "model" / args.model_file)
    obs_npz = load_npz(case_dir / "data" / "waveform" / "obs_data.npz")
    obs_subset = subset_obs_npz(obs_npz, shot_count=args.shot_count, nt_samples=args.nt_samples)

    survey = build_survey(obs_subset, f0=args.f0)
    observed_forward_seconds = None
    if args.observed_source == "saved":
        obs_data = build_observed_data(survey, obs_npz, shot_count=args.shot_count, nt_samples=args.nt_samples)
    elif args.observed_source == "synthetic-true":
        true_model_npz = load_npz(case_dir / "data" / "model" / "true_model.npz")
        obs_data, observed_forward_seconds = synthesize_observed_data(true_model_npz, survey, args, backend)
    else:
        raise ValueError(f"unsupported observed source: {args.observed_source}")
    model = build_inversion_model(model_npz, args)
    initial_vp = model.vp.detach().clone()
    propagator = AcousticPropagator(model, survey)

    grad_mask = np.ones((model.nz, model.nx), dtype=np.float32)
    if args.grad_mute_top > 0:
        grad_mask[: args.grad_mute_top, :] = 0.0
    gradient_processor = build_gradient_processor(args, grad_mask)

    optimizer, optimizer_name = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    loss_fn, loss_name = build_loss(args)

    fwi = AcousticFWI(
        propagator,
        model,
        optimizer,
        scheduler,
        loss_fn,
        obs_data,
        gradient_processor=gradient_processor,
        waveform_normalize=args.waveform_normalize,
        cache_result=True,
        cache_result_epoch=1,
        save_fig_epoch=-1,
    )

    progress_context = nullcontext() if args.show_progress else redirect_stderr(io.StringIO())
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with progress_context:
        fwi.forward(iteration=args.iterations, batch_size=args.shot_count, checkpoint_segments=args.checkpoint_segments)
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    seconds = time.perf_counter() - start

    if not fwi.iter_loss:
        raise RuntimeError("AcousticFWI did not record an iteration loss")
    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after reduced Marmousi2 inversion")

    losses = [float(value) for value in fwi.iter_loss]
    if len(losses) < args.iterations:
        raise RuntimeError(f"expected at least {args.iterations} recorded losses, got {len(losses)}")
    for idx, value in enumerate(losses):
        finite_value(value, f"loss[{idx}]")

    loss_summary = summarize_losses(losses)
    loss = loss_summary["loss"]
    grad_norm = tensor_norm(model.vp.grad)
    update_norm = tensor_norm(model.vp.detach() - initial_vp)
    finite_value(loss, "loss")
    finite_positive(grad_norm, "vp_grad_norm")
    finite_positive(update_norm, "vp_update_norm")

    report = {
        "status": "ok",
        "case": "Marmousi2 acoustic reduced inversion",
        "case_dir": str(case_dir),
        "backend": ADFWI.backend_diagnostics(),
        "subset": {
            "shot_count": args.shot_count,
            "nt_samples": args.nt_samples,
            "checkpoint_segments": args.checkpoint_segments,
            "receivers": survey.receiver.num,
            "dt": survey.source.dt,
        },
        "observed": {
            "source": args.observed_source,
            "synthetic_true_forward_seconds": observed_forward_seconds,
        },
        "model": {
            "vp": tensor_summary(model.vp),
            "rho": tensor_summary(model.rho),
            "nx": model.nx,
            "nz": model.nz,
            "dx": model.dx,
            "dz": model.dz,
            "nabc": model.nabc,
        },
        "inversion": {
            "iterations": args.iterations,
            "optimizer": optimizer_name,
            "lr": args.lr,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "misfit": loss_name,
            "waveform_normalize": args.waveform_normalize,
            "auto_update_rho": args.auto_update_rho,
            "gradient_processor": args.gradient_processor,
            "norm_grad": args.norm_grad,
            "forw_illumination": args.forw_illumination,
            "grad_mute": args.grad_mute,
            "grad_smooth": args.grad_smooth,
            "grad_mute_top": args.grad_mute_top,
            "marine_or_land": args.marine_or_land,
            "seconds": seconds,
            **loss_summary,
            "vp_grad_norm": grad_norm,
            "vp_update_norm": update_norm,
        },
        "seed": args.seed,
    }
    if args.output_dir is not None:
        report["outputs"] = write_outputs(args.output_dir, report, initial_vp, model.vp.detach(), losses)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--model-file", default="init_model.npz", choices=("init_model.npz", "true_model.npz"))
    parser.add_argument("--device", default="auto", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority used when --device auto")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", type=parse_dtype, default="float32")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--f0", type=float, default=5.0)
    parser.add_argument("--abc-type", default="PML")
    parser.add_argument("--abc-jerjan-alpha", type=float, default=0.007)
    parser.add_argument("--shot-count", type=int, default=1)
    parser.add_argument("--nt-samples", type=int, default=300)
    parser.add_argument("--observed-source", choices=("saved", "synthetic-true"), default="saved")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--lr", type=float, default=1e12)
    parser.add_argument("--scheduler-step-size", type=int, default=1)
    parser.add_argument("--scheduler-gamma", type=float, default=1.0)
    parser.add_argument("--dt-for-loss", type=float, default=1.0)
    parser.add_argument("--misfit", default="safe-squared-l2", choices=("safe-squared-l2", "legacy-l2"))
    parser.add_argument("--grad-mute-top", type=int, default=12)
    parser.add_argument("--grad-mute", type=int, default=0)
    parser.add_argument("--grad-smooth", type=int, default=0)
    parser.add_argument("--marine-or-land", choices=("marine", "offshore", "land", "onshore"), default="land")
    parser.add_argument("--gradient-processor", choices=("legacy", "torch"), default="legacy")
    parser.add_argument("--norm-grad", action="store_true")
    parser.add_argument("--forw-illumination", action="store_true")
    parser.add_argument("--auto-update-rho", action="store_true")
    parser.add_argument("--waveform-normalize", action="store_true")
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-dir", type=Path, help="optional directory for JSON, CSV, and PNG outputs")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
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
