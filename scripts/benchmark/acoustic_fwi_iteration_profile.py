#!/usr/bin/env python
"""Profile one Marmousi2 acoustic FWI iteration.

This benchmark is the Phase A gate for propagator performance work. It does not
change ADFWI runtime behavior. It reproduces a validation inversion setup, then
times one or more explicit FWI iterations split into forward, loss, backward,
gradient processing, optimizer, and setup costs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_CASES = {
    "reduced": {
        "case": "marmousi2_acoustic_reduced",
        "scripts": REPO_ROOT / "examples" / "validation" / "marmousi2_acoustic_reduced" / "scripts",
        "output_root": REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_reduced"
        / "outputs"
        / "phase_a_profile",
    },
    "full_record": {
        "case": "marmousi2_acoustic_full_record",
        "scripts": REPO_ROOT / "examples" / "validation" / "marmousi2_acoustic_full_record" / "scripts",
        "output_root": REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_full_record"
        / "outputs"
        / "phase_a_profile",
    },
}
DEFAULT_RESULT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_fwi_iteration_profile_20260531.json"
)
forward_modeling = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name!r} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_validation_case(case_name: str):
    global forward_modeling
    case = VALIDATION_CASES[case_name]
    forward_modeling = load_module(f"phase_a_{case_name}_forward_modeling", case["scripts"] / "forward_modeling.py")
    return case


def add_arguments(parser: argparse.ArgumentParser, validation_case: str) -> None:
    case = configure_validation_case(validation_case)
    parser.add_argument("--validation-case", choices=tuple(VALIDATION_CASES), default=validation_case)
    forward_modeling.add_case_arguments(parser)
    parser.set_defaults(output_root=case["output_root"])
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--scheduler-step-size", type=int, default=200)
    parser.add_argument("--scheduler-gamma", type=float, default=0.75)
    parser.add_argument("--grad-mute-top", type=int, default=12)
    parser.add_argument("--gaussian-kernel", type=int, default=6)
    parser.add_argument("--rcv-depth", type=int, default=10)
    parser.add_argument("--mask-extra-depth", type=int, default=2)
    parser.add_argument("--generate-observed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-forward-wavefield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-forw-illumination", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--storage-policy",
        choices=("pressure_remat",),
        default=None,
        help="Expert opt-in AcousticPropagator storage/replay policy. Default keeps the production path.",
    )
    parser.add_argument(
        "--pressure-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Acoustic pressure-only policy. Default None lets AcousticFWI use its auto policy.",
    )
    parser.add_argument("--gradient-processor", choices=("legacy", "torch"), default="legacy")
    parser.add_argument("--policy-repeat", type=int, default=1)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="FWI shot batch size. Defaults to --shots so existing runs keep full-batch behavior.",
    )


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def memory_api(torch_module, backend):
    if backend.name == "npu":
        return torch_module.npu
    if backend.name == "cuda":
        return torch_module.cuda
    return None


def reset_peak_memory(torch_module, backend) -> None:
    api = memory_api(torch_module, backend)
    if api is None:
        return
    if hasattr(api, "empty_cache"):
        api.empty_cache()
    if hasattr(api, "reset_peak_memory_stats"):
        api.reset_peak_memory_stats()
    elif hasattr(api, "reset_max_memory_allocated"):
        api.reset_max_memory_allocated()


def max_memory_allocated(torch_module, backend):
    api = memory_api(torch_module, backend)
    if api is None or not hasattr(api, "max_memory_allocated"):
        return None
    return int(api.max_memory_allocated())


class Timer:
    def __init__(self, backend) -> None:
        self.backend = backend

    def measure(self, fn):
        synchronize(self.backend)
        start = time.perf_counter()
        value = fn()
        synchronize(self.backend)
        return value, time.perf_counter() - start


def tensor_scalar(value) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu().item())
    return float(value)


def tensor_norm(value) -> float:
    torch = sys.modules["torch"]
    detached = value.detach()
    return float(torch.linalg.norm(detached.reshape(-1)).cpu().item())


def build_initial_model(rt: Dict[str, Any], args: argparse.Namespace):
    np = rt["np"]
    torch = rt["torch"]
    true_model, vp_true, _ = forward_modeling.build_true_arrays(rt, args)
    from ADFWI.utils import get_smooth_marmousi_model

    smooth_model = get_smooth_marmousi_model(
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
        dtype=torch.float32 if args.dtype == "float32" else torch.float64,
    )
    return model, vp_init


def ensure_observed_data(rt: Dict[str, Any], args: argparse.Namespace, backend) -> Dict[str, Any]:
    obs_path = args.output_root / "waveform" / "obs_data.npz"
    if obs_path.exists():
        return {"generated": False, "path": str(obs_path), "seconds": 0.0}
    if not args.generate_observed:
        raise FileNotFoundError(f"missing observed data: {obs_path}")

    timer = Timer(backend)
    _, seconds = timer.measure(lambda: forward_modeling.run_forward(args))
    return {"generated": True, "path": str(obs_path), "seconds": seconds}


def build_fwi_state(rt: Dict[str, Any], args: argparse.Namespace, backend):
    import numpy as np
    import torch

    from ADFWI.fwi import AcousticFWI
    from ADFWI.fwi.misfit import Misfit_waveform_L2
    from ADFWI.propagator import GradProcessor, TorchGradProcessor

    model, vp_init = build_initial_model(rt, args)
    survey = forward_modeling.build_survey(rt, args)
    propagator = rt["AcousticPropagator"](model, survey, device=args.device)

    d_obs = rt["SeismicData"](survey)
    d_obs.load(str(args.output_root / "waveform" / "obs_data.npz"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step_size,
        gamma=args.scheduler_gamma,
        last_epoch=-1,
    )
    grad_mask = np.ones_like(vp_init)
    grad_mask[: args.grad_mute_top, :] = 0
    processor_cls = TorchGradProcessor if args.gradient_processor == "torch" else GradProcessor
    gradient_processor = processor_cls(
        grad_mask=grad_mask,
        forw_illumination=args.grad_forw_illumination,
    )

    fwi = AcousticFWI(
        propagator=propagator,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=Misfit_waveform_L2(dt=1),
        obs_data=d_obs,
        gradient_processor=gradient_processor,
        waveform_normalize=True,
        cache_result=False,
        save_fig_epoch=-1,
    )
    fwi._validate_forward_wavefield_policy(args.save_forward_wavefield)
    return fwi, vp_init


def tensor_diff(reference, candidate, *, atol_floor=1e-12) -> Dict[str, Any]:
    torch = sys.modules["torch"]
    ref = reference.detach().cpu()
    val = candidate.detach().cpu()
    diff = (val - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(ref, atol_floor))
    rel = diff / denom
    return {
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "reference_norm": float(torch.linalg.norm(ref.reshape(-1)).item()),
        "candidate_norm": float(torch.linalg.norm(val.reshape(-1)).item()),
    }


def public_iteration_report(report: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def run_one_iteration(fwi, args: argparse.Namespace, timer: Timer, *, keep_tensors: bool = False) -> Dict[str, Any]:
    import torch

    from ADFWI.fwi.acoustic_fwi import acoustic_gradient_parameter_specs
    from ADFWI.fwi.iteration.batches import iter_batch_ranges
    from ADFWI.fwi.iteration.epoch import apply_epoch_update_step
    from ADFWI.fwi.iteration.loss import acoustic_pressure_loss_input, build_batch_loss, evaluate_loss_inputs
    from ADFWI.fwi.runtime.forward import acoustic_forward_batch
    from ADFWI.fwi.runtime.gradient import process_named_parameter_gradients
    from ADFWI.fwi.runtime.wavefield import acoustic_pressure_waveforms, accumulate_wavefield

    timings = {
        "zero_grad": 0.0,
        "forward": 0.0,
        "wavefield_accumulation": 0.0,
        "loss_evaluation": 0.0,
        "backward": 0.0,
        "gradient_processing": 0.0,
        "optimizer_step": 0.0,
    }
    batch_reports: List[Dict[str, Any]] = []

    _, timings["zero_grad"] = timer.measure(lambda: fwi.optimizer.zero_grad())
    loss_epoch = 0.0
    accumulated_wavefield = None
    batch_size = args.batch_size if args.batch_size is not None else args.shots
    batch_ranges = list(iter_batch_ranges(fwi.propagator.src_n, batch_size))

    for batch_range in batch_ranges:
        forward_batch, elapsed = timer.measure(
            lambda batch_range=batch_range: acoustic_forward_batch(
                fwi.propagator,
                batch_range,
                args.checkpoint_segments,
                save_forward_wavefield=args.save_forward_wavefield,
                storage_policy=args.storage_policy,
                pressure_only="auto" if args.pressure_only is None else args.pressure_only,
            )
        )
        timings["forward"] += elapsed

        (_, forward_wavefield_p), elapsed = timer.measure(
            lambda: acoustic_pressure_waveforms(forward_batch.record_waveform)
        )
        timings["wavefield_accumulation"] += elapsed
        accumulated_wavefield, elapsed = timer.measure(
            lambda: accumulate_wavefield(accumulated_wavefield, forward_wavefield_p)
        )
        timings["wavefield_accumulation"] += elapsed

        def evaluate_loss():
            loss_input = acoustic_pressure_loss_input(
                forward_batch.record_waveform,
                fwi.obs_p,
                forward_batch.shot_index,
            )
            loss_evaluation = evaluate_loss_inputs(
                [loss_input],
                prepare_loss_pair=fwi._prepare_loss_pair,
                loss_fn=fwi.loss_fn,
                normalization=fwi.waveform_normalize,
                function_fallback="apply",
                cutoff_freq=None,
                propagator_dt=fwi.propagator.dt,
                device=fwi.device,
            )
            return build_batch_loss(loss_evaluation.data_loss)

        batch_loss, elapsed = timer.measure(evaluate_loss)
        timings["loss_evaluation"] += elapsed
        loss_epoch += batch_loss.scalar

        _, elapsed = timer.measure(lambda batch_loss=batch_loss: batch_loss.tensor.backward())
        timings["backward"] += elapsed
        batch_reports.append(
            {
                "batch": batch_range.batch,
                "begin": batch_range.begin,
                "end": batch_range.end,
                "loss": batch_loss.scalar,
            }
        )

    raw_grad = fwi.model.vp.grad.detach().clone()
    raw_grad_norm = tensor_norm(raw_grad)
    raw_grad_finite = bool(torch.isfinite(raw_grad).all().cpu().item())

    _, timings["gradient_processing"] = timer.measure(
        lambda: process_named_parameter_gradients(
            fwi.model,
            acoustic_gradient_parameter_specs(),
            fwi.process_gradient,
            forw=accumulated_wavefield,
        )
    )
    grad_after_processing = fwi.model.vp.grad.detach()
    processed_grad = grad_after_processing.clone()
    grad_norm_after_processing = tensor_norm(grad_after_processing)
    grad_finite_after_processing = bool(torch.isfinite(grad_after_processing).all().cpu().item())

    _, timings["optimizer_step"] = timer.measure(
        lambda: apply_epoch_update_step(fwi.optimizer, fwi.scheduler, fwi.model)
    )
    vp_finite_after_optimizer = bool(torch.isfinite(fwi.model.vp.detach()).all().cpu().item())
    vp_after_optimizer = fwi.model.vp.detach().clone()

    total = sum(timings.values())
    report = {
        "loss": float(loss_epoch),
        "loss_finite": bool(torch.isfinite(torch.as_tensor(loss_epoch)).item()),
        "timings": timings,
        "timing_total": total,
        "timing_fraction": {name: (value / total if total > 0 else 0.0) for name, value in timings.items()},
        "raw_grad_norm": raw_grad_norm,
        "raw_grad_finite": raw_grad_finite,
        "grad_norm_after_processing": grad_norm_after_processing,
        "grad_finite_after_processing": grad_finite_after_processing,
        "vp_finite_after_optimizer": vp_finite_after_optimizer,
        "batches": batch_reports,
    }
    if keep_tensors:
        report["_raw_grad"] = raw_grad
        report["_processed_grad"] = processed_grad
        report["_vp_after_optimizer"] = vp_after_optimizer
    return report


def run_profile(args: argparse.Namespace) -> Dict[str, Any]:
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    setup_start = time.perf_counter()
    rt = forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    observed_report = ensure_observed_data(rt, args, backend)
    fwi, vp_init = build_fwi_state(rt, args, backend)
    reset_peak_memory(rt["torch"], backend)
    synchronize(backend)
    setup_seconds = time.perf_counter() - setup_start

    timer = Timer(backend)
    iteration_reports = [
        public_iteration_report(run_one_iteration(fwi, args, timer))
        for _ in range(args.iterations)
    ]

    torch = rt["torch"]
    vp_update_norm = float(
        torch.linalg.norm(
            (fwi.model.vp.detach() - torch.as_tensor(vp_init, device=fwi.model.vp.device, dtype=fwi.model.vp.dtype)).reshape(-1)
        )
        .cpu()
        .item()
    )
    peak_allocated = max_memory_allocated(rt["torch"], backend)

    return {
        "status": "ok",
        "case": VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Phase A end-to-end acoustic FWI iteration cost breakdown",
        "backend": rt["ADFWI"].backend_diagnostics(),
        "shape": {
            "shots": args.shots,
            "batch_size": args.batch_size if args.batch_size is not None else args.shots,
            "receivers": fwi.propagator.rcv_n,
            "nt": fwi.propagator.nt,
            "nx": fwi.model.nx,
            "nz": fwi.model.nz,
            "checkpoint_segments": args.checkpoint_segments,
            "save_forward_wavefield": args.save_forward_wavefield,
            "grad_forw_illumination": args.grad_forw_illumination,
            "storage_policy": args.storage_policy,
            "pressure_only": "auto" if args.pressure_only is None else args.pressure_only,
            "gradient_processor": args.gradient_processor,
        },
        "setup_seconds": setup_seconds,
        "observed_data": observed_report,
        "iterations": iteration_reports,
        "vp_update_norm": vp_update_norm,
        "memory": {
            "peak_allocated_bytes": peak_allocated,
            "peak_allocated_mib": (
                peak_allocated / (1024.0 * 1024.0)
                if peak_allocated is not None
                else None
            ),
        },
    }


def run_policy_variant(args: argparse.Namespace, *, save_forward_wavefield: bool):
    variant_args = argparse.Namespace(**vars(args))
    variant_args.save_forward_wavefield = save_forward_wavefield
    variant_args.grad_forw_illumination = False
    setup_start = time.perf_counter()
    rt = forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(variant_args.device, dtype=variant_args.dtype, fallback=variant_args.fallback_cpu)
    observed_report = ensure_observed_data(rt, variant_args, backend)
    fwi, vp_init = build_fwi_state(rt, variant_args, backend)
    synchronize(backend)
    setup_seconds = time.perf_counter() - setup_start

    timer = Timer(backend)
    iteration = run_one_iteration(fwi, variant_args, timer, keep_tensors=True)
    torch = rt["torch"]
    vp_update_norm = float(
        torch.linalg.norm(
            (fwi.model.vp.detach() - torch.as_tensor(vp_init, device=fwi.model.vp.device, dtype=fwi.model.vp.dtype)).reshape(-1)
        )
        .cpu()
        .item()
    )
    return {
        "backend": rt["ADFWI"].backend_diagnostics(),
        "setup_seconds": setup_seconds,
        "observed_data": observed_report,
        "shape": {
            "shots": variant_args.shots,
            "batch_size": variant_args.batch_size if variant_args.batch_size is not None else variant_args.shots,
            "receivers": fwi.propagator.rcv_n,
            "nt": fwi.propagator.nt,
            "nx": fwi.model.nx,
            "nz": fwi.model.nz,
            "checkpoint_segments": variant_args.checkpoint_segments,
            "save_forward_wavefield": save_forward_wavefield,
            "grad_forw_illumination": False,
        },
        "iteration": iteration,
        "vp_update_norm": vp_update_norm,
    }


def run_policy_pair(args: argparse.Namespace, *, reverse_order: bool = False) -> Dict[str, Any]:
    if reverse_order:
        skipped = run_policy_variant(args, save_forward_wavefield=False)
        full = run_policy_variant(args, save_forward_wavefield=True)
    else:
        full = run_policy_variant(args, save_forward_wavefield=True)
        skipped = run_policy_variant(args, save_forward_wavefield=False)

    full_iter = full["iteration"]
    skipped_iter = skipped["iteration"]
    comparison = {
        "loss_abs_diff": abs(skipped_iter["loss"] - full_iter["loss"]),
        "raw_grad": tensor_diff(full_iter["_raw_grad"], skipped_iter["_raw_grad"]),
        "processed_grad": tensor_diff(full_iter["_processed_grad"], skipped_iter["_processed_grad"]),
        "vp_after_optimizer": tensor_diff(full_iter["_vp_after_optimizer"], skipped_iter["_vp_after_optimizer"]),
        "speedup": {
            "forward": full_iter["timings"]["forward"] / skipped_iter["timings"]["forward"],
            "backward": full_iter["timings"]["backward"] / skipped_iter["timings"]["backward"],
            "total": full_iter["timing_total"] / skipped_iter["timing_total"],
        },
    }
    return {
        "order": "candidate_first" if reverse_order else "reference_first",
        "reference": {
            **{key: value for key, value in full.items() if key != "iteration"},
            "iteration": public_iteration_report(full_iter),
        },
        "candidate": {
            **{key: value for key, value in skipped.items() if key != "iteration"},
            "iteration": public_iteration_report(skipped_iter),
        },
        "comparison": comparison,
    }


def summarize_policy_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def values(path):
        result = []
        for pair in pairs:
            current = pair
            for key in path:
                current = current[key]
            result.append(float(current))
        return result

    def stats(path):
        series = values(path)
        return {
            "min": min(series),
            "max": max(series),
            "mean": sum(series) / len(series),
            "values": series,
        }

    return {
        "speedup": {
            "forward": stats(("comparison", "speedup", "forward")),
            "backward": stats(("comparison", "speedup", "backward")),
            "total": stats(("comparison", "speedup", "total")),
        },
        "timing_seconds": {
            "reference_total": stats(("reference", "iteration", "timing_total")),
            "candidate_total": stats(("candidate", "iteration", "timing_total")),
            "reference_forward": stats(("reference", "iteration", "timings", "forward")),
            "candidate_forward": stats(("candidate", "iteration", "timings", "forward")),
            "reference_backward": stats(("reference", "iteration", "timings", "backward")),
            "candidate_backward": stats(("candidate", "iteration", "timings", "backward")),
        },
        "max_differences": {
            "loss_abs_diff": max(values(("comparison", "loss_abs_diff"))),
            "raw_grad_max_abs_diff": max(values(("comparison", "raw_grad", "max_abs_diff"))),
            "raw_grad_max_rel_diff": max(values(("comparison", "raw_grad", "max_rel_diff"))),
            "processed_grad_max_abs_diff": max(values(("comparison", "processed_grad", "max_abs_diff"))),
            "processed_grad_max_rel_diff": max(values(("comparison", "processed_grad", "max_rel_diff"))),
            "vp_after_optimizer_max_abs_diff": max(values(("comparison", "vp_after_optimizer", "max_abs_diff"))),
            "vp_after_optimizer_max_rel_diff": max(values(("comparison", "vp_after_optimizer", "max_rel_diff"))),
        },
    }


def run_policy_comparison(args: argparse.Namespace) -> Dict[str, Any]:
    if args.grad_forw_illumination:
        raise ValueError("--compare-wavefield-policy requires --no-grad-forw-illumination")
    if args.policy_repeat <= 0:
        raise ValueError("--policy-repeat must be positive")

    pairs = [
        run_policy_pair(args, reverse_order=bool(index % 2))
        for index in range(args.policy_repeat)
    ]

    return {
        "status": "ok",
        "case": VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Phase B acoustic FWI forward-wavefield policy comparison",
        "policy_repeat": args.policy_repeat,
        "pairs": pairs,
        "summary": summarize_policy_pairs(pairs),
    }


def main(argv: Optional[List[str]] = None) -> int:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)

    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, pre_args.validation_case)
    parser.add_argument("--compare-wavefield-policy", action="store_true")
    args = parser.parse_args(argv)
    forward_modeling.validate_case_args(parser, args)

    if args.compare_wavefield_policy:
        report = run_policy_comparison(args)
    else:
        report = run_profile(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=forward_modeling.json_default) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=forward_modeling.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
