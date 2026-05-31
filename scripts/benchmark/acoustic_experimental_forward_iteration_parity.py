#!/usr/bin/env python
"""Compare one acoustic FWI-style iteration with experimental forward path.

This benchmark keeps production ADFWI code unchanged. It builds the reduced
Marmousi2 validation setup, runs one differentiable pressure-loss iteration
with production ``AcousticPropagator.forward`` and once with the benchmark-only
``experimental_forward_kernel``, then compares receiver records, loss, and raw
``vp.grad``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward import (
    experimental_chunk_forward_kernel,
    experimental_forward_kernel,
)
from ADFWI.fwi.runtime.forward import ForwardBatchRecord


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_experimental_forward_iteration_parity_20260531.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


class Timer:
    def __init__(self, backend) -> None:
        self.backend = backend

    def measure(self, fn):
        synchronize(self.backend)
        start = time.perf_counter()
        value = fn()
        synchronize(self.backend)
        return value, time.perf_counter() - start


def experimental_forward_batch(propagator, batch_range, *, save_forward_wavefield: bool, mode: str):
    propagator.model.forward()
    shot_index = batch_range.shot_index
    if mode == "experimental":
        forward_kernel = experimental_forward_kernel
    elif mode == "experimental-chunk":
        forward_kernel = experimental_chunk_forward_kernel
    else:
        raise ValueError(f"unknown experimental mode: {mode}")
    record_waveform = forward_kernel(
        propagator.nx,
        propagator.nz,
        propagator.dx,
        propagator.dz,
        propagator.nt,
        propagator.dt,
        propagator.nabc,
        propagator.free_surface,
        propagator.src_x[shot_index],
        propagator.src_z[shot_index],
        len(shot_index),
        propagator.wavelet[shot_index],
        propagator.rcv_x,
        propagator.rcv_z,
        propagator.rcv_n,
        propagator.damp,
        propagator.model.vp,
        propagator.model.rho,
        save_forward_wavefield=save_forward_wavefield,
        device=propagator.device,
        dtype=propagator.dtype,
    )
    return ForwardBatchRecord(shot_index=shot_index, record_waveform=record_waveform)


def run_iteration(fwi, args: argparse.Namespace, timer: Timer, *, mode: str) -> Dict[str, Any]:
    import torch

    from ADFWI.fwi.iteration.batches import iter_batch_ranges
    from ADFWI.fwi.iteration.loss import acoustic_pressure_loss_input, build_batch_loss, evaluate_loss_inputs
    from ADFWI.fwi.runtime.forward import acoustic_forward_batch

    timings = {"zero_grad": 0.0, "forward": 0.0, "loss_evaluation": 0.0, "backward": 0.0}
    outputs = []
    losses = []

    _, timings["zero_grad"] = timer.measure(lambda: fwi.optimizer.zero_grad())
    for batch_range in iter_batch_ranges(fwi.propagator.src_n, args.batch_size):
        if mode == "production":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: acoustic_forward_batch(
                    fwi.propagator,
                    batch_range,
                    args.checkpoint_segments,
                    save_forward_wavefield=False,
                )
            )
        elif mode == "production-custom-chunk":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: acoustic_forward_batch(
                    fwi.propagator,
                    batch_range,
                    args.checkpoint_segments,
                    save_forward_wavefield=False,
                    use_custom_chunk_backward=True,
                )
            )
        elif mode == "experimental":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: experimental_forward_batch(
                    fwi.propagator,
                    batch_range,
                    save_forward_wavefield=False,
                    mode=mode,
                )
            )
        elif mode == "experimental-chunk":
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: experimental_forward_batch(
                    fwi.propagator,
                    batch_range,
                    save_forward_wavefield=False,
                    mode=mode,
                )
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        timings["forward"] += elapsed
        outputs.append({name: value.detach().clone() for name, value in forward_batch.record_waveform.items() if name in {"p", "u", "w"}})

        def evaluate_loss():
            if args.loss_mode == "synthetic-energy":
                tensor = sum(
                    forward_batch.record_waveform[name].pow(2).mean()
                    for name in ("p", "u", "w")
                )
                return type("SyntheticEnergyLoss", (), {"tensor": tensor, "scalar": float(tensor.detach().cpu().item())})()
            if args.loss_mode != "observed-pressure":
                raise ValueError(f"unknown loss mode: {args.loss_mode}")
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
        losses.append(batch_loss)
        _, elapsed = timer.measure(lambda batch_loss=batch_loss: batch_loss.tensor.backward())
        timings["backward"] += elapsed

    if fwi.model.vp.grad is None:
        raise RuntimeError(f"{mode} vp.grad is None")
    raw_grad = fwi.model.vp.grad.detach().clone()
    loss_epoch = float(sum(item.scalar for item in losses))
    return {
        "loss": loss_epoch,
        "outputs": outputs,
        "raw_grad": raw_grad,
        "raw_grad_finite": bool(torch.isfinite(raw_grad).all().cpu().item()),
        "timings": timings,
        "timing_total": sum(timings.values()),
    }


def run_variant(args: argparse.Namespace, *, mode: str) -> Dict[str, Any]:
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    profile.ensure_observed_data(rt, args, backend)
    fwi, _ = profile.build_fwi_state(rt, args, backend)
    fwi._validate_forward_wavefield_policy(False)
    fwi.waveform_normalize = args.waveform_normalize
    timer = Timer(backend)
    result = run_iteration(fwi, args, timer, mode=mode)
    return {
        "backend": rt["ADFWI"].backend_diagnostics(),
        "shape": {
            "shots": fwi.propagator.src_n,
            "batch_size": args.batch_size,
            "receivers": fwi.propagator.rcv_n,
            "nt": fwi.propagator.nt,
            "nx": fwi.model.nx,
            "nz": fwi.model.nz,
            "checkpoint_segments": args.checkpoint_segments,
        },
        "iteration": result,
    }


def compare(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    output_diffs = []
    for reference_batch, candidate_batch in zip(reference["iteration"]["outputs"], candidate["iteration"]["outputs"]):
        output_diffs.append({
            name: profile.tensor_diff(reference_batch[name], candidate_batch[name])
            for name in ("p", "u", "w")
        })
    return {
        "loss_abs_diff": abs(candidate["iteration"]["loss"] - reference["iteration"]["loss"]),
        "outputs": output_diffs,
        "raw_grad": profile.tensor_diff(reference["iteration"]["raw_grad"], candidate["iteration"]["raw_grad"]),
        "speedup": {
            "forward": reference["iteration"]["timings"]["forward"] / candidate["iteration"]["timings"]["forward"],
            "backward": reference["iteration"]["timings"]["backward"] / candidate["iteration"]["timings"]["backward"],
            "total": reference["iteration"]["timing_total"] / candidate["iteration"]["timing_total"],
        },
    }


def public_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    iteration = variant["iteration"]
    return {
        "backend": variant["backend"],
        "shape": variant["shape"],
        "iteration": {
            "loss": iteration["loss"],
            "raw_grad_finite": iteration["raw_grad_finite"],
            "timings": iteration["timings"],
            "timing_total": iteration["timing_total"],
        },
    }


def summarize_pair(pair: Dict[str, Any]) -> Dict[str, Any]:
    output_max_abs = max(
        diff["max_abs_diff"]
        for batch in pair["comparison"]["outputs"]
        for diff in batch.values()
    )
    output_max_rel = max(
        diff["max_rel_diff"]
        for batch in pair["comparison"]["outputs"]
        for diff in batch.values()
    )
    return {
        "loss_abs_diff": pair["comparison"]["loss_abs_diff"],
        "output_max_abs_diff": output_max_abs,
        "output_max_rel_diff": output_max_rel,
        "raw_grad_max_abs_diff": pair["comparison"]["raw_grad"]["max_abs_diff"],
        "raw_grad_max_rel_diff": pair["comparison"]["raw_grad"]["max_rel_diff"],
        "speedup": pair["comparison"]["speedup"],
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    profile.ensure_observed_data(rt, args, backend)
    reference = run_variant(args, mode="production")
    candidate = run_variant(args, mode=args.candidate_mode)
    pair = {
        "reference": public_variant(reference),
        "candidate": public_variant(candidate),
        "comparison": compare(reference, candidate),
    }
    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": f"Reduced Marmousi2 production vs {args.candidate_mode} acoustic forward iteration parity",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pair": pair,
        "summary": summarize_pair(pair),
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument(
        "--waveform-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply the normal FWI waveform normalization before loss evaluation.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=("experimental", "experimental-chunk", "production", "production-custom-chunk"),
        default="experimental",
        help="Compare production against an experimental path or a second production run.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("observed-pressure", "synthetic-energy"),
        default="observed-pressure",
        help="Use normal observed-data pressure loss or direct synthetic energy loss.",
    )
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        output_root=REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_reduced"
        / "outputs"
        / "experimental_forward_iteration_parity",
        shots=3,
        batch_size=3,
        nx=64,
        nz=32,
        nt=120,
        save_forward_wavefield=False,
        grad_forw_illumination=False,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.save_forward_wavefield:
        parser.error("experimental iteration parity requires --no-save-forward-wavefield")
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
