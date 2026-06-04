#!/usr/bin/env python
"""Break down acoustic one-iteration memory by forward/loss/backward phase.

This is a diagnostic benchmark for the acoustic rematerialized pressure path.
It does not edit production kernels.  The goal is to determine whether the
full-batch peak memory is caused by forward cache retention, loss graph
construction, backward replay, or optimizer-side work.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import acoustic_experimental_forward_iteration_parity as parity
from scripts.benchmark import acoustic_fwi_iteration_profile as profile


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_memory_phase_breakdown_20260604.json"
)


def memory_allocated(torch_module, backend) -> Optional[int]:
    api = parity.memory_api(torch_module, backend)
    if api is None or not hasattr(api, "memory_allocated"):
        return None
    return int(api.memory_allocated())


def memory_snapshot(torch_module, backend) -> Dict[str, Optional[float]]:
    return {
        "current_allocated_mib": parity.bytes_to_mib(memory_allocated(torch_module, backend)),
        "peak_allocated_mib": parity.bytes_to_mib(parity.max_memory_allocated(torch_module, backend)),
    }


def measure_phase(torch_module, backend, name: str, fn):
    parity.synchronize(backend)
    parity.reset_peak_memory(torch_module, backend)
    before = memory_snapshot(torch_module, backend)
    start = time.perf_counter()
    value = fn()
    parity.synchronize(backend)
    elapsed = time.perf_counter() - start
    after = memory_snapshot(torch_module, backend)
    return value, {
        "phase": name,
        "seconds": elapsed,
        "before": before,
        "after": after,
    }


def forward_batch_for_mode(fwi, args: argparse.Namespace, batch_range):
    from ADFWI.fwi.runtime.forward import acoustic_forward_batch

    if args.mode == "production":
        return acoustic_forward_batch(
            fwi.propagator,
            batch_range,
            args.checkpoint_segments,
            save_forward_wavefield=args.save_forward_wavefield,
            pressure_only=args.pressure_only,
        )
    if args.mode == "production-remat-pressure-stride2":
        return acoustic_forward_batch(
            fwi.propagator,
            batch_range,
            args.checkpoint_segments,
            save_forward_wavefield=False,
            custom_chunk_strategy="remat_pressure_stride2",
            pressure_only=True,
        )
    if args.mode == "experimental-remat-pressure-chunk":
        return parity.experimental_forward_batch(
            fwi.propagator,
            batch_range,
            save_forward_wavefield=False,
            mode=args.mode,
            checkpoint_segments=args.checkpoint_segments,
            remat_divergence_cache_stride=args.remat_divergence_cache_stride,
            remat_divergence_cache_components=args.remat_divergence_cache_components,
        )
    raise ValueError(f"unsupported mode: {args.mode}")


def build_batch_loss(fwi, args: argparse.Namespace, forward_batch):
    from ADFWI.fwi.iteration.loss import acoustic_pressure_loss_input, build_batch_loss, evaluate_loss_inputs

    if args.loss_mode == "synthetic-energy":
        tensor = sum(forward_batch.record_waveform[name].pow(2).mean() for name in ("p", "u", "w"))
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


def run_mode(args: argparse.Namespace, *, mode: str) -> Dict[str, Any]:
    from ADFWI.fwi.acoustic_fwi import acoustic_gradient_parameter_specs
    from ADFWI.fwi.iteration.batches import iter_batch_ranges
    from ADFWI.fwi.runtime.gradient import process_named_parameter_gradients

    mode_args = argparse.Namespace(**vars(args))
    mode_args.mode = mode
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(mode_args.device, dtype=mode_args.dtype, fallback=mode_args.fallback_cpu)
    profile.ensure_observed_data(rt, mode_args, backend)
    fwi, _ = profile.build_fwi_state(rt, mode_args, backend)
    fwi._validate_forward_wavefield_policy(mode_args.save_forward_wavefield)
    torch = rt["torch"]

    batch_size = mode_args.batch_size if mode_args.batch_size is not None else mode_args.shots
    batch_ranges = list(iter_batch_ranges(fwi.propagator.src_n, batch_size))
    if len(batch_ranges) != 1:
        raise ValueError("phase breakdown expects one full batch; set --batch-size equal to --shots")

    phases = []
    parity.synchronize(backend)
    _, phase = measure_phase(torch, backend, "zero_grad", lambda: fwi.optimizer.zero_grad())
    phases.append(phase)

    forward_batch, phase = measure_phase(
        torch,
        backend,
        "forward",
        lambda: forward_batch_for_mode(fwi, mode_args, batch_ranges[0]),
    )
    phases.append(phase)

    batch_loss, phase = measure_phase(torch, backend, "loss_evaluation", lambda: build_batch_loss(fwi, mode_args, forward_batch))
    phases.append(phase)

    _, phase = measure_phase(torch, backend, "backward", lambda: batch_loss.tensor.backward())
    phases.append(phase)

    raw_grad = fwi.model.vp.grad.detach()
    raw_grad_finite = bool(torch.isfinite(raw_grad).all().cpu().item())
    raw_grad_norm = profile.tensor_norm(raw_grad)

    _, phase = measure_phase(
        torch,
        backend,
        "gradient_processing",
        lambda: process_named_parameter_gradients(
            fwi.model,
            acoustic_gradient_parameter_specs(),
            fwi.process_gradient,
            forw=None,
        ),
    )
    phases.append(phase)

    return {
        "mode": mode,
        "backend": rt["ADFWI"].backend_diagnostics(),
        "shape": {
            "shots": mode_args.shots,
            "batch_size": batch_size,
            "receivers": fwi.propagator.rcv_n,
            "nt": fwi.propagator.nt,
            "nx": fwi.model.nx,
            "nz": fwi.model.nz,
            "checkpoint_segments": mode_args.checkpoint_segments,
            "loss_mode": mode_args.loss_mode,
            "save_forward_wavefield": mode_args.save_forward_wavefield,
            "remat_divergence_cache_stride": mode_args.remat_divergence_cache_stride,
            "remat_divergence_cache_components": mode_args.remat_divergence_cache_components,
        },
        "loss": float(batch_loss.scalar),
        "loss_finite": bool(torch.isfinite(batch_loss.tensor.detach()).cpu().item()),
        "raw_grad_finite": raw_grad_finite,
        "raw_grad_norm": raw_grad_norm,
        "phases": phases,
    }


def parse_modes(value: str) -> list[str]:
    modes = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"production", "production-remat-pressure-stride2", "experimental-remat-pressure-chunk"}
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown mode(s): {unknown}; allowed: {sorted(allowed)}")
    return modes


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="full_record")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument(
        "--modes",
        type=parse_modes,
        default=parse_modes("production,production-remat-pressure-stride2"),
        help="Comma separated modes to profile.",
    )
    parser.add_argument(
        "--loss-mode",
        choices=("observed-pressure", "synthetic-energy"),
        default="observed-pressure",
    )
    parser.add_argument(
        "--waveform-normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--remat-divergence-cache-stride", type=int, default=2)
    parser.add_argument("--remat-divergence-cache-components", default="p,u,w")
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        output_root=REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_full_record"
        / "outputs"
        / "memory_phase_breakdown",
        shots=40,
        batch_size=40,
        nx=200,
        nz=88,
        nt=3000,
        checkpoint_segments=10,
        save_forward_wavefield=False,
        grad_forw_illumination=False,
        pressure_only=True,
    )
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    reports = [run_mode(args, mode=mode) for mode in args.modes]
    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Acoustic one-iteration phase-level memory breakdown",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reports": reports,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.save_forward_wavefield:
        parser.error("memory phase breakdown requires --no-save-forward-wavefield")
    if args.batch_size != args.shots:
        parser.error("memory phase breakdown currently expects --batch-size equal to --shots")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(json.dumps(report["reports"], indent=2, sort_keys=True))
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
