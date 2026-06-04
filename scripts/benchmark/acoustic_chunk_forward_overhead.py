#!/usr/bin/env python
"""Profile forward-only overhead for acoustic custom-gradient candidates.

This benchmark keeps production propagator code unchanged. It compares the
production acoustic forward path against the benchmark-only timestep-custom and
chunk-custom paths on the same validation geometry, using receiver outputs as
the parity contract.
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

from ADFWI.fwi.runtime.forward import acoustic_forward_batch
from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward_iteration_parity import experimental_forward_batch


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_chunk_forward_overhead_20260531.json"
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


def run_forward_mode(fwi, args: argparse.Namespace, timer: Timer, *, mode: str) -> Dict[str, Any]:
    from ADFWI.fwi.iteration.batches import iter_batch_ranges

    outputs = []
    elapsed_total = 0.0
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
        elif mode in {"experimental", "experimental-chunk"}:
            forward_batch, elapsed = timer.measure(
                lambda batch_range=batch_range: experimental_forward_batch(
                    fwi.propagator,
                    batch_range,
                    save_forward_wavefield=False,
                    mode=mode,
                )
            )
        elif mode == "experimental-pressure-remat":
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
        elapsed_total += elapsed
        outputs.append(
            {
                name: value.detach().clone()
                for name, value in forward_batch.record_waveform.items()
                if name in {"p", "u", "w"}
            }
        )
    return {"seconds": elapsed_total, "outputs": outputs}


def run_once(args: argparse.Namespace, *, mode: str) -> Dict[str, Any]:
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    profile.ensure_observed_data(rt, args, backend)
    fwi, _ = profile.build_fwi_state(rt, args, backend)
    fwi._validate_forward_wavefield_policy(False)
    timer = Timer(backend)
    result = run_forward_mode(fwi, args, timer, mode=mode)
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
        "forward_seconds": result["seconds"],
        "outputs": result["outputs"],
    }


def compare_outputs(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    batches = []
    for reference_batch, candidate_batch in zip(reference["outputs"], candidate["outputs"]):
        batches.append(
            {
                name: profile.tensor_diff(reference_batch[name], candidate_batch[name])
                for name in ("p", "u", "w")
            }
        )
    return {
        "batches": batches,
        "max_abs_diff": max(diff["max_abs_diff"] for batch in batches for diff in batch.values()),
        "max_rel_diff": max(diff["max_rel_diff"] for batch in batches for diff in batch.values()),
    }


def public_variant(variant: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in variant.items() if key != "outputs"}


def run(args: argparse.Namespace) -> Dict[str, Any]:
    requested_modes = tuple(item.strip() for item in args.modes.split(",") if item.strip())
    if "production" not in requested_modes:
        raise ValueError("--modes must include production")

    variants = {mode: [] for mode in requested_modes}
    comparisons = []
    for index in range(args.warmup + args.repeat):
        run_outputs = {mode: run_once(args, mode=mode) for mode in requested_modes}
        if index < args.warmup:
            continue
        production = run_outputs["production"]
        public_outputs = {mode: public_variant(value) for mode, value in run_outputs.items()}
        speedup = {
            mode: production["forward_seconds"] / value["forward_seconds"]
            for mode, value in run_outputs.items()
            if mode != "production"
        }
        output_diff = {
            mode: compare_outputs(production, value)
            for mode, value in run_outputs.items()
            if mode != "production"
        }
        comparisons.append({"index": index - args.warmup, "speedup": speedup, "output_diff": output_diff})
        for mode, value in public_outputs.items():
            variants[mode].append(value)

    def stats(values):
        return {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
            "values": values,
        }

    summary = {
        "forward_seconds": {
            mode: stats([item["forward_seconds"] for item in items])
            for mode, items in variants.items()
        },
        "speedup_vs_production": {
            mode: stats([comparison["speedup"][mode] for comparison in comparisons])
            for mode in requested_modes
            if mode != "production"
        },
        "output_max_abs_diff": {
            mode: max(comparison["output_diff"][mode]["max_abs_diff"] for comparison in comparisons)
            for mode in requested_modes
            if mode != "production"
        },
        "output_max_rel_diff": {
            mode: max(comparison["output_diff"][mode]["max_rel_diff"] for comparison in comparisons)
            for mode in requested_modes
            if mode != "production"
        },
    }
    return {
        "status": "ok",
        "purpose": "forward-only overhead profile for acoustic custom-gradient candidates",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "variants": variants,
        "comparisons": comparisons,
        "summary": summary,
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument(
        "--modes",
        default="production,experimental,experimental-chunk",
        help="Comma-separated forward modes to compare. Must include production.",
    )
    parser.add_argument(
        "--remat-divergence-cache-stride",
        type=int,
        default=0,
        help="For rematerialized modes: 0 disables divergence caching; N caches every Nth step.",
    )
    parser.add_argument(
        "--remat-divergence-cache-components",
        default="p,u,w",
        help="For rematerialized modes: comma/space separated subset of p,u,w divergence terms.",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        output_root=REPO_ROOT
        / "examples"
        / "validation"
        / "marmousi2_acoustic_reduced"
        / "outputs"
        / "experimental_forward_iteration_parity_fullshape",
        shots=3,
        nx=200,
        nz=88,
        nt=3000,
        save_forward_wavefield=False,
        grad_forw_illumination=False,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
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
