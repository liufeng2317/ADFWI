#!/usr/bin/env python
"""Compare production and experimental acoustic forward paths in an FWI loop.

This benchmark is intentionally outside the production runtime. It answers one
question: does an experimental forward path that passes one-iteration parity
also preserve the loss trajectory and speed behavior across a short optimizer
loop?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward_iteration_parity import (
    experimental_forward_batch,
    max_memory_allocated,
    reset_peak_memory,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_experimental_fwi_loop_compare_20260601.json"
)


def run_one_iteration(fwi, args: argparse.Namespace, timer: profile.Timer, *, mode: str) -> Dict[str, Any]:
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
    loss_epoch = 0.0
    accumulated_wavefield = None
    batch_size = args.batch_size if args.batch_size is not None else args.shots

    _, timings["zero_grad"] = timer.measure(lambda: fwi.optimizer.zero_grad())
    for batch_range in iter_batch_ranges(fwi.propagator.src_n, batch_size):
        if mode == "production":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: acoustic_forward_batch(
                    fwi.propagator,
                    batch_range,
                    args.checkpoint_segments,
                    save_forward_wavefield=args.save_forward_wavefield,
                )
            )
        elif mode == "production-remat-pressure-stride2":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: acoustic_forward_batch(
                    fwi.propagator,
                    batch_range,
                    args.checkpoint_segments,
                    save_forward_wavefield=False,
                    custom_chunk_strategy="remat_pressure_stride2",
                    pressure_only=True,
                )
            )
        elif mode == "experimental-remat-pressure-chunk":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: experimental_forward_batch(
                    fwi.propagator,
                    batch_range,
                    save_forward_wavefield=False,
                    mode=mode,
                    checkpoint_segments=args.checkpoint_segments,
                    remat_divergence_cache_stride=args.remat_divergence_cache_stride,
                    remat_divergence_cache_components=args.remat_divergence_cache_components,
                )
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
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

    raw_grad = fwi.model.vp.grad.detach().clone()
    _, timings["gradient_processing"] = timer.measure(
        lambda: process_named_parameter_gradients(
            fwi.model,
            acoustic_gradient_parameter_specs(),
            fwi.process_gradient,
            forw=accumulated_wavefield,
        )
    )
    processed_grad = fwi.model.vp.grad.detach().clone()
    _, timings["optimizer_step"] = timer.measure(
        lambda: apply_epoch_update_step(fwi.optimizer, fwi.scheduler, fwi.model)
    )
    return {
        "loss": float(loss_epoch),
        "loss_finite": bool(torch.isfinite(torch.as_tensor(loss_epoch)).cpu().item()),
        "raw_grad_finite": bool(torch.isfinite(raw_grad).all().cpu().item()),
        "processed_grad_finite": bool(torch.isfinite(processed_grad).all().cpu().item()),
        "raw_grad_norm": profile.tensor_norm(raw_grad),
        "processed_grad_norm": profile.tensor_norm(processed_grad),
        "timings": timings,
        "timing_total": sum(timings.values()),
    }


def run_variant(args: argparse.Namespace, *, mode: str) -> Dict[str, Any]:
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    profile.ensure_observed_data(rt, args, backend)
    fwi, vp_init = profile.build_fwi_state(rt, args, backend)
    fwi._validate_forward_wavefield_policy(args.save_forward_wavefield)
    torch = rt["torch"]
    reset_peak_memory(torch, backend)
    profile.synchronize(backend)
    timer = profile.Timer(backend)
    iterations = [run_one_iteration(fwi, args, timer, mode=mode) for _ in range(args.iterations)]
    profile.synchronize(backend)
    peak_memory = max_memory_allocated(torch, backend)
    vp_update_norm = float(
        torch.linalg.norm(
            (fwi.model.vp.detach() - torch.as_tensor(vp_init, device=fwi.model.vp.device, dtype=fwi.model.vp.dtype)).reshape(-1)
        )
        .cpu()
        .item()
    )
    return {
        "mode": mode,
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
        },
        "iterations": iterations,
        "vp_update_norm": vp_update_norm,
        "memory": {
            "peak_allocated_bytes": peak_memory,
            "peak_allocated_mib": peak_memory / (1024.0 * 1024.0) if peak_memory is not None else None,
        },
    }


def summarize_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    iterations = variant["iterations"]
    return {
        "losses": [item["loss"] for item in iterations],
        "total_seconds": sum(item["timing_total"] for item in iterations),
        "mean_seconds_per_iteration": sum(item["timing_total"] for item in iterations) / len(iterations),
        "forward_seconds": sum(item["timings"]["forward"] for item in iterations),
        "backward_seconds": sum(item["timings"]["backward"] for item in iterations),
        "gradient_processing_seconds": sum(item["timings"]["gradient_processing"] for item in iterations),
        "all_finite": all(
            item["loss_finite"] and item["raw_grad_finite"] and item["processed_grad_finite"]
            for item in iterations
        ),
        "vp_update_norm": variant["vp_update_norm"],
        "peak_allocated_mib": variant["memory"]["peak_allocated_mib"],
    }


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    ref = summarize_variant(reference)
    cand = summarize_variant(candidate)
    loss_abs_diffs = [abs(a - b) for a, b in zip(ref["losses"], cand["losses"])]
    return {
        "reference": ref,
        "candidate": cand,
        "loss_abs_diffs": loss_abs_diffs,
        "max_loss_abs_diff": max(loss_abs_diffs),
        "vp_update_norm_abs_diff": abs(cand["vp_update_norm"] - ref["vp_update_norm"]),
        "speedup": {
            "total": ref["total_seconds"] / cand["total_seconds"],
            "mean_iteration": ref["mean_seconds_per_iteration"] / cand["mean_seconds_per_iteration"],
            "forward": ref["forward_seconds"] / cand["forward_seconds"],
            "backward": ref["backward_seconds"] / cand["backward_seconds"],
        },
        "memory": {
            "reference_peak_allocated_mib": ref["peak_allocated_mib"],
            "candidate_peak_allocated_mib": cand["peak_allocated_mib"],
            "candidate_over_reference_peak_allocated": (
                cand["peak_allocated_mib"] / ref["peak_allocated_mib"]
                if ref["peak_allocated_mib"] else None
            ),
        },
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    reference = run_variant(args, mode="production")
    candidate = run_variant(args, mode=args.candidate_mode)
    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": f"Short FWI loop comparison: production vs {args.candidate_mode}",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reference": reference,
        "candidate": candidate,
        "summary": compare(reference, candidate),
    }


def build_parser(argv: Optional[List[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument(
        "--candidate-mode",
        choices=(
            "production-remat-pressure-stride2",
            "experimental-remat-pressure-chunk",
        ),
        default="production-remat-pressure-stride2",
    )
    parser.add_argument("--remat-divergence-cache-stride", type=int, default=1)
    parser.add_argument("--remat-divergence-cache-components", default="p,u,w")
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        iterations=5,
        output_root=REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_reduced"
        / "outputs"
        / "experimental_fwi_loop_compare",
        shots=3,
        batch_size=3,
        nx=200,
        nz=88,
        nt=3000,
        checkpoint_segments=10,
        save_forward_wavefield=False,
        grad_forw_illumination=False,
        gradient_processor="legacy",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.save_forward_wavefield:
        parser.error("experimental FWI loop comparison requires --no-save-forward-wavefield")
    if args.grad_forw_illumination:
        parser.error("experimental FWI loop comparison requires --no-grad-forw-illumination")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
