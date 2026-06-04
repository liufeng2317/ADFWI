#!/usr/bin/env python
"""Probe observed-pressure upstream gradients for acoustic experimental path.

The previous parity runs showed exact receiver outputs and loss, but different
raw ``vp.grad``. This script separates two questions:

1. Does observed-pressure loss produce the same upstream gradient with respect
   to receiver records?
2. If the same upstream gradient is applied directly to receiver records, do
   production autograd and the custom backward produce the same ``vp.grad``?
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
from ADFWI.fwi.iteration.loss import acoustic_pressure_loss_input, build_batch_loss, evaluate_loss_inputs
from ADFWI.fwi.runtime.forward import acoustic_forward_batch
from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward_iteration_parity import experimental_forward_batch


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_observed_loss_upstream_probe_20260531.json"
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


def run_forward(fwi, args: argparse.Namespace, timer: Timer, *, mode: str):
    batch_range = next(iter_batch_ranges(fwi.propagator.src_n, args.batch_size))
    if mode == "production":
        return timer.measure(
            lambda: acoustic_forward_batch(
                fwi.propagator,
                batch_range,
                args.checkpoint_segments,
                save_forward_wavefield=False,
            )
        )
    if mode == "experimental":
        return timer.measure(
            lambda: experimental_forward_batch(
                fwi.propagator,
                batch_range,
                save_forward_wavefield=False,
            )
        )
    raise ValueError(f"unknown mode: {mode}")


def retain_receiver_grads(record_waveform: Dict[str, Any]) -> None:
    for component in ("p", "u", "w"):
        record_waveform[component].retain_grad()


def receiver_grads(record_waveform: Dict[str, Any]) -> Dict[str, Any]:
    torch = sys.modules["torch"]
    grads = {}
    for component in ("p", "u", "w"):
        grad = record_waveform[component].grad
        if grad is None:
            grad = torch.zeros_like(record_waveform[component])
        grads[component] = grad.detach().clone()
    return grads


def observed_loss(fwi, forward_batch, args: argparse.Namespace):
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


def run_observed_variant(args: argparse.Namespace, *, mode: str) -> Dict[str, Any]:
    rt, backend, fwi = build_fwi(args)
    timer = Timer(backend)
    _, zero_seconds = timer.measure(lambda: fwi.optimizer.zero_grad())
    forward_batch, forward_seconds = run_forward(fwi, args, timer, mode=mode)
    retain_receiver_grads(forward_batch.record_waveform)
    batch_loss, loss_seconds = timer.measure(lambda: observed_loss(fwi, forward_batch, args))
    _, backward_seconds = timer.measure(lambda: batch_loss.tensor.backward())
    return {
        "backend": rt["ADFWI"].backend_diagnostics(),
        "loss": batch_loss.scalar,
        "outputs": {
            component: forward_batch.record_waveform[component].detach().clone()
            for component in ("p", "u", "w")
        },
        "receiver_grads": receiver_grads(forward_batch.record_waveform),
        "raw_grad": fwi.model.vp.grad.detach().clone(),
        "timings": {
            "zero_grad": zero_seconds,
            "forward": forward_seconds,
            "loss": loss_seconds,
            "backward": backward_seconds,
        },
    }


def external_loss(record_waveform: Dict[str, Any], upstream: Dict[str, Any]):
    return sum(
        (record_waveform[component] * upstream[component].to(record_waveform[component].device)).sum()
        for component in ("p", "u", "w")
    )


def run_external_variant(args: argparse.Namespace, *, mode: str, upstream: Dict[str, Any]) -> Dict[str, Any]:
    rt, backend, fwi = build_fwi(args)
    timer = Timer(backend)
    _, zero_seconds = timer.measure(lambda: fwi.optimizer.zero_grad())
    forward_batch, forward_seconds = run_forward(fwi, args, timer, mode=mode)
    tensor, loss_seconds = timer.measure(lambda: external_loss(forward_batch.record_waveform, upstream))
    _, backward_seconds = timer.measure(lambda: tensor.backward())
    return {
        "backend": rt["ADFWI"].backend_diagnostics(),
        "loss": float(tensor.detach().cpu().item()),
        "outputs": {
            component: forward_batch.record_waveform[component].detach().clone()
            for component in ("p", "u", "w")
        },
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


def compare_outputs(reference: Dict[str, Any], candidate: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {
        component: tensor_diff(reference[key][component], candidate[key][component])
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


def run(args: argparse.Namespace) -> Dict[str, Any]:
    production_observed = run_observed_variant(args, mode="production")
    experimental_observed = run_observed_variant(args, mode="experimental")
    production_external = run_external_variant(args, mode="production", upstream=production_observed["receiver_grads"])
    experimental_external = run_external_variant(args, mode="experimental", upstream=production_observed["receiver_grads"])

    observed_output = compare_outputs(production_observed, experimental_observed, "outputs")
    observed_upstream = compare_outputs(production_observed, experimental_observed, "receiver_grads")
    external_output = compare_outputs(production_external, experimental_external, "outputs")

    return {
        "status": "ok",
        "purpose": "Observed-pressure upstream-gradient and external-gradient parity probe",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "observed_loss": {
            "production": public_variant(production_observed),
            "experimental": public_variant(experimental_observed),
            "loss_abs_diff": abs(experimental_observed["loss"] - production_observed["loss"]),
            "outputs": observed_output,
            "receiver_upstream_grads": observed_upstream,
            "raw_grad": tensor_diff(production_observed["raw_grad"], experimental_observed["raw_grad"]),
        },
        "external_upstream": {
            "production": public_variant(production_external),
            "experimental": public_variant(experimental_external),
            "loss_abs_diff": abs(experimental_external["loss"] - production_external["loss"]),
            "outputs": external_output,
            "raw_grad": tensor_diff(production_external["raw_grad"], experimental_external["raw_grad"]),
        },
        "summary": {
            "observed_loss_abs_diff": abs(experimental_observed["loss"] - production_observed["loss"]),
            "observed_output": summarize_component_diffs(observed_output),
            "observed_receiver_upstream": summarize_component_diffs(observed_upstream),
            "observed_raw_grad": tensor_diff(production_observed["raw_grad"], experimental_observed["raw_grad"]),
            "external_loss_abs_diff": abs(experimental_external["loss"] - production_external["loss"]),
            "external_output": summarize_component_diffs(external_output),
            "external_raw_grad": tensor_diff(production_external["raw_grad"], experimental_external["raw_grad"]),
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
