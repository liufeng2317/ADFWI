#!/usr/bin/env python
"""Notebook-equivalent Marmousi2 acoustic inversion validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from forward_modeling import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    REPO_ROOT,
    add_case_arguments,
    build_survey,
    build_true_arrays,
    import_runtime_modules,
    json_default,
    validate_case_args,
    write_json,
)


def add_inversion_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--scheduler-step-size", type=int, default=200)
    parser.add_argument("--scheduler-gamma", type=float, default=0.75)
    parser.add_argument("--grad-mute-top", type=int, default=12)
    parser.add_argument("--gaussian-kernel", type=int, default=6)
    parser.add_argument("--rcv-depth", type=int, default=10)
    parser.add_argument("--mask-extra-depth", type=int, default=2)


def ensure_inversion_dirs(output_root: Path) -> None:
    for subdir in ("model", "waveform", "survey", "inversion"):
        (output_root / subdir).mkdir(parents=True, exist_ok=True)


def import_inversion_runtime_modules() -> Dict[str, Any]:
    rt = import_runtime_modules()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ADFWI.fwi import AcousticFWI
    from ADFWI.fwi.misfit import Misfit_global_correlation, Misfit_waveform_L2
    from ADFWI.propagator import GradProcessor
    from ADFWI.utils import get_smooth_marmousi_model

    rt.update(
        {
            "plt": plt,
            "AcousticFWI": AcousticFWI,
            "Misfit_global_correlation": Misfit_global_correlation,
            "Misfit_waveform_L2": Misfit_waveform_L2,
            "GradProcessor": GradProcessor,
            "get_smooth_marmousi_model": get_smooth_marmousi_model,
        }
    )
    return rt


def build_initial_model(rt: Dict[str, Any], args: argparse.Namespace):
    np = rt["np"]
    true_model, vp_true, _ = build_true_arrays(rt, args)
    smooth_model = rt["get_smooth_marmousi_model"](
        true_model,
        gaussian_kernel=args.gaussian_kernel,
        rcv_depth=args.rcv_depth,
        mask_extra_detph=args.mask_extra_depth,
    )
    vp_init = smooth_model["vp"].T
    rho_init = np.power(vp_init, 0.25) * 310
    model = rt["AcousticModel"](
        args.ox,
        args.oz,
        args.nx,
        args.nz,
        args.dx,
        args.dz,
        vp_init,
        rho_init,
        vp_bound=[vp_true.min(), vp_true.max()],
        vp_grad=True,
        free_surface=args.free_surface,
        abc_type=args.abc_type,
        abc_jerjan_alpha=args.abc_jerjan_alpha,
        nabc=args.nabc,
        auto_update_rho=True,
        device=args.device,
        dtype=rt["torch"].float32 if args.dtype == "float32" else rt["torch"].float64,
    )
    return model, vp_init, vp_true


def save_inversion_figures(rt: Dict[str, Any], output_root: Path, vp_init, vp_true, iter_vp, iter_loss) -> None:
    plt = rt["plt"]
    final_vp = iter_vp[-1]

    plt.figure(figsize=(8, 6))
    plt.plot(iter_loss, c="k")
    plt.xlabel("Iteration")
    plt.ylabel("Misfit")
    plt.grid(True, alpha=0.3)
    plt.savefig(output_root / "inversion" / "loss.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(vp_init, cmap="coolwarm")
    plt.title("Initial vp")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(final_vp, cmap="coolwarm")
    plt.title("Inverted vp")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(output_root / "inversion" / "init_vs_inverted_vp.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(vp_true, cmap="coolwarm")
    plt.title("True vp")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.imshow(final_vp, cmap="coolwarm")
    plt.title("Inverted vp")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(output_root / "inversion" / "true_vs_inverted_vp.png", bbox_inches="tight")
    plt.close()


def run_inversion(args: argparse.Namespace, *, iterations: Optional[int] = None) -> Dict[str, Any]:
    rt = import_inversion_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    np = rt["np"]
    torch = rt["torch"]
    ensure_inversion_dirs(args.output_root)

    obs_path = args.output_root / "waveform" / "obs_data.npz"
    if not obs_path.exists():
        raise FileNotFoundError(f"missing observed data: {obs_path}; run forward_modeling.py first")

    iteration_count = args.iterations if iterations is None else iterations
    model, vp_init, vp_true = build_initial_model(rt, args)
    model.save(str(args.output_root / "model" / "init_model.npz"))
    model._plot_vp_rho(
        figsize=(12, 5),
        wspace=0.15,
        cbar_pad_fraction=0.01,
        cmap="coolwarm",
        save_path=str(args.output_root / "model" / "init_vp_rho.png"),
    )

    survey = build_survey(rt, args)
    survey.plot(model.vp, cmap="coolwarm", save_path=str(args.output_root / "survey" / "observed_system_init.png"))
    propagator = rt["AcousticPropagator"](model, survey, device=args.device)
    rt["plot_damp"](propagator.damp, save_path=str(args.output_root / "model" / "boundary_condition.png"))

    d_obs = rt["SeismicData"](survey)
    d_obs.load(str(obs_path))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
        last_epoch=-1,
    )
    loss_fn = rt["Misfit_waveform_L2"](dt=1)
    grad_mask = np.ones_like(vp_init)
    grad_mask[: args.grad_mute_top, :] = 0
    gradient_processor = rt["GradProcessor"](grad_mask=grad_mask)

    fwi = rt["AcousticFWI"](
        propagator=propagator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        obs_data=d_obs,
        gradient_processor=gradient_processor,
        waveform_normalize=True,
        cache_result=True,
        save_fig_epoch=50,
        save_fig_path=str(args.output_root / "inversion"),
    )

    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    fwi.forward(iteration=iteration_count, batch_size=args.shots, checkpoint_segments=args.checkpoint_segments)
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    seconds = time.perf_counter() - start

    iter_vp = fwi.iter_vp
    iter_loss = [float(value) for value in fwi.iter_loss]
    np.savez(args.output_root / "inversion" / "iter_vp.npz", data=np.array(iter_vp))
    np.savez(args.output_root / "inversion" / "iter_loss.npz", data=np.array(iter_loss))
    save_inversion_figures(rt, args.output_root, vp_init, vp_true, iter_vp, iter_loss)

    report = {
        "status": "ok",
        "stage": "inversion",
        "output_root": str(args.output_root),
        "obs_data": str(obs_path),
        "backend": rt["ADFWI"].backend_diagnostics(),
        "iterations": iteration_count,
        "shots": args.shots,
        "checkpoint_segments": args.checkpoint_segments,
        "seconds": seconds,
        "initial_loss": iter_loss[0],
        "final_loss": iter_loss[-1],
        "loss_history": iter_loss,
        "vp_update_norm": float(
            torch.linalg.norm(
                (model.vp.detach() - torch.as_tensor(vp_init, device=model.vp.device, dtype=model.vp.dtype)).reshape(-1)
            )
            .cpu()
            .item()
        ),
    }
    write_json(args.output_root / "inversion" / f"inversion{iteration_count}_summary.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_case_arguments(parser)
    add_inversion_arguments(parser)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_case_args(parser, args)
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    try:
        report = run_inversion(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "stage": "inversion", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
