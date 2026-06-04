#!/usr/bin/env python
"""Replay observed-pressure upstream gradients directly through acoustic kernels.

This is a targeted validation probe. It first captures the exact receiver
upstream gradient produced by the reduced Marmousi2 observed-pressure loss, then
applies that upstream as an external linear loss to the production
``forward_kernel`` and the benchmark-only experimental custom-gradient kernel.

The goal is to decide whether the remaining ``vp.grad`` mismatch is caused by
the observed upstream distribution itself, or by the surrounding FWI wrapper /
model parameter dependency path.
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

from ADFWI.fwi.iteration.batches import iter_batch_ranges
from ADFWI.propagator.acoustic_kernels import forward_kernel
from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward import experimental_forward_kernel
from scripts.benchmark.acoustic_observed_loss_upstream_probe import run_observed_variant


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_observed_upstream_direct_replay_20260531.json"
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


def build_fwi(args: argparse.Namespace):
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    profile.ensure_observed_data(rt, args, backend)
    fwi, _ = profile.build_fwi_state(rt, args, backend)
    fwi._validate_forward_wavefield_policy(False)
    fwi.waveform_normalize = args.waveform_normalize
    return rt, backend, fwi


def select_first_batch(propagator, args: argparse.Namespace):
    return next(iter_batch_ranges(propagator.src_n, args.batch_size))


def run_production_direct(fwi, args: argparse.Namespace):
    propagator = fwi.propagator
    batch_range = select_first_batch(propagator, args)
    shot_index = batch_range.shot_index
    propagator.model.forward()
    return forward_kernel(
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
        checkpoint_segments=args.checkpoint_segments,
        save_forward_wavefield=False,
        device=propagator.device,
        dtype=propagator.dtype,
    )


def run_experimental_direct(fwi, args: argparse.Namespace):
    propagator = fwi.propagator
    batch_range = select_first_batch(propagator, args)
    shot_index = batch_range.shot_index
    propagator.model.forward()
    return experimental_forward_kernel(
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
        save_forward_wavefield=False,
        device=propagator.device,
        dtype=propagator.dtype,
    )


def external_receiver_loss(record_waveform: Dict[str, Any], upstream: Dict[str, Any]):
    return sum(
        (record_waveform[component] * upstream[component].to(record_waveform[component].device)).sum()
        for component in ("p", "u", "w")
    )


def run_direct_variant(args: argparse.Namespace, *, mode: str, upstream: Dict[str, Any]) -> Dict[str, Any]:
    rt, backend, fwi = build_fwi(args)
    timer = Timer(backend)
    _, zero_seconds = timer.measure(lambda: fwi.optimizer.zero_grad())
    if mode == "production":
        record, forward_seconds = timer.measure(lambda: run_production_direct(fwi, args))
    elif mode == "experimental":
        record, forward_seconds = timer.measure(lambda: run_experimental_direct(fwi, args))
    else:
        raise ValueError(f"unknown mode: {mode}")
    loss, loss_seconds = timer.measure(lambda: external_receiver_loss(record, upstream))
    _, backward_seconds = timer.measure(lambda: loss.backward())
    if fwi.model.vp.grad is None:
        raise RuntimeError(f"{mode} vp.grad is None")
    return {
        "backend": rt["ADFWI"].backend_diagnostics(),
        "loss": float(loss.detach().cpu().item()),
        "outputs": {component: record[component].detach().clone() for component in ("p", "u", "w")},
        "raw_grad": fwi.model.vp.grad.detach().clone(),
        "timings": {
            "zero_grad": zero_seconds,
            "forward": forward_seconds,
            "loss": loss_seconds,
            "backward": backward_seconds,
        },
    }


def tensor_diff(reference, candidate) -> Dict[str, Any]:
    return profile.tensor_diff(reference, candidate)


def compare_outputs(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        component: tensor_diff(reference["outputs"][component], candidate["outputs"][component])
        for component in ("p", "u", "w")
    }


def summarize_component_diffs(diffs: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    return {
        "max_abs_diff": max(item["max_abs_diff"] for item in diffs.values()),
        "max_rel_diff": max(item["max_rel_diff"] for item in diffs.values()),
    }


def public_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backend": variant["backend"],
        "loss": variant["loss"],
        "timings": variant["timings"],
    }


def upstream_summary(upstream: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    torch = sys.modules["torch"]
    summary = {}
    for component, value in upstream.items():
        detached = value.detach()
        nonzero = detached != 0
        summary[component] = {
            "shape": list(detached.shape),
            "max_abs": float(detached.abs().max().cpu().item()),
            "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
            "nonzero_count": int(nonzero.sum().cpu().item()),
        }
    return summary


def run(args: argparse.Namespace) -> Dict[str, Any]:
    observed = run_observed_variant(args, mode="production")
    upstream = observed["receiver_grads"]
    production = run_direct_variant(args, mode="production", upstream=upstream)
    experimental = run_direct_variant(args, mode="experimental", upstream=upstream)
    output_diff = compare_outputs(production, experimental)
    raw_grad_diff = tensor_diff(production["raw_grad"], experimental["raw_grad"])
    return {
        "status": "ok",
        "purpose": "Direct replay of observed-pressure receiver upstream through production and experimental acoustic kernels",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "observed_upstream_source": public_variant(observed),
        "upstream": upstream_summary(upstream),
        "direct_replay": {
            "production": public_variant(production),
            "experimental": public_variant(experimental),
            "loss_abs_diff": abs(experimental["loss"] - production["loss"]),
            "outputs": output_diff,
            "raw_grad": raw_grad_diff,
        },
        "summary": {
            "loss_abs_diff": abs(experimental["loss"] - production["loss"]),
            "output": summarize_component_diffs(output_diff),
            "raw_grad": raw_grad_diff,
        },
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument(
        "--waveform-normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        output_root=REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_reduced"
        / "outputs"
        / "observed_upstream_direct_replay",
        shots=3,
        nx=200,
        nz=88,
        nt=3000,
        nabc=30,
        save_forward_wavefield=False,
        grad_forw_illumination=False,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True, default=profile.forward_modeling.json_default))
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
