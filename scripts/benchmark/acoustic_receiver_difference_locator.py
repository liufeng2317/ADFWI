#!/usr/bin/env python
"""Locate receiver-output differences for the acoustic experimental path.

This script compares production ``AcousticPropagator.forward`` against the
benchmark-only ``experimental_forward_kernel`` under the validation
source/wavelet setup. It reports where receiver-record differences first
appear and where they accumulate.
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
from ADFWI.fwi.runtime.forward import acoustic_forward_batch
from scripts.benchmark import acoustic_fwi_iteration_profile as profile
from scripts.benchmark.acoustic_experimental_forward_iteration_parity import experimental_forward_batch


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_receiver_difference_locator_20260531.json"
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


def build_propagator(args: argparse.Namespace):
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    model, _ = profile.build_initial_model(rt, args)
    survey = profile.forward_modeling.build_survey(rt, args)
    propagator = rt["AcousticPropagator"](model, survey, device=args.device)
    return rt, backend, propagator


def run_forward_pair(args: argparse.Namespace):
    rt, backend, propagator = build_propagator(args)
    timer = Timer(backend)
    batch_range = next(iter_batch_ranges(propagator.src_n, args.batch_size))
    production, production_seconds = timer.measure(
        lambda: acoustic_forward_batch(
            propagator,
            batch_range,
            args.checkpoint_segments,
            save_forward_wavefield=False,
        )
    )
    experimental, experimental_seconds = timer.measure(
        lambda: experimental_forward_batch(
            propagator,
            batch_range,
            save_forward_wavefield=False,
        )
    )
    return {
        "rt": rt,
        "backend": backend,
        "propagator": propagator,
        "batch_range": batch_range,
        "production": production.record_waveform,
        "experimental": experimental.record_waveform,
        "timings": {
            "production_forward": production_seconds,
            "experimental_forward": experimental_seconds,
            "speedup": production_seconds / experimental_seconds if experimental_seconds > 0 else None,
        },
    }


def unravel_index(flat_index: int, shape):
    indices = []
    for size in reversed(shape):
        indices.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(indices))


def scalar(value) -> float:
    return float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)


def index_record(tensor, index) -> float:
    return scalar(tensor[index])


def topk_records(diff, rel, reference, candidate, *, top_k: int):
    torch = sys.modules["torch"]
    flat = diff.reshape(-1)
    count = min(top_k, flat.numel())
    values, indices = torch.topk(flat, k=count)
    records = []
    for value, flat_index_tensor in zip(values, indices):
        flat_index = int(flat_index_tensor.cpu().item())
        index = unravel_index(flat_index, diff.shape)
        records.append(
            {
                "shot": int(index[0]),
                "time": int(index[1]),
                "receiver": int(index[2]),
                "abs_diff": scalar(value),
                "rel_diff": index_record(rel, index),
                "reference": index_record(reference, index),
                "candidate": index_record(candidate, index),
            }
        )
    return records


def first_diff_record(diff, rel, reference, candidate, *, threshold: float):
    torch = sys.modules["torch"]
    locations = torch.nonzero(diff > threshold, as_tuple=False)
    if locations.numel() == 0:
        return None
    index = tuple(int(item) for item in locations[0].cpu().tolist())
    return {
        "shot": int(index[0]),
        "time": int(index[1]),
        "receiver": int(index[2]),
        "abs_diff": index_record(diff, index),
        "rel_diff": index_record(rel, index),
        "reference": index_record(reference, index),
        "candidate": index_record(candidate, index),
    }


def per_shot_records(diff, rel):
    torch = sys.modules["torch"]
    records = []
    for shot in range(diff.shape[0]):
        shot_diff = diff[shot]
        flat_index = int(torch.argmax(shot_diff.reshape(-1)).cpu().item())
        time_index, receiver_index = unravel_index(flat_index, shot_diff.shape)
        records.append(
            {
                "shot": shot,
                "max_abs_diff": index_record(shot_diff, (time_index, receiver_index)),
                "max_rel_diff": index_record(rel[shot], (time_index, receiver_index)),
                "time": int(time_index),
                "receiver": int(receiver_index),
            }
        )
    return records


def time_distribution(diff, *, top_k: int):
    torch = sys.modules["torch"]
    per_time_max = diff.amax(dim=(0, 2))
    per_time_sum = diff.sum(dim=(0, 2))
    count = min(top_k, per_time_max.numel())
    max_values, max_indices = torch.topk(per_time_max, k=count)
    sum_values, sum_indices = torch.topk(per_time_sum, k=count)
    first_nonzero = torch.nonzero(per_time_max > 0.0, as_tuple=False)
    return {
        "first_nonzero_time": None if first_nonzero.numel() == 0 else int(first_nonzero[0].cpu().item()),
        "top_time_by_max_abs": [
            {"time": int(index.cpu().item()), "max_abs_diff": scalar(value)}
            for value, index in zip(max_values, max_indices)
        ],
        "top_time_by_sum_abs": [
            {"time": int(index.cpu().item()), "sum_abs_diff": scalar(value)}
            for value, index in zip(sum_values, sum_indices)
        ],
    }


def component_report(reference, candidate, args: argparse.Namespace) -> Dict[str, Any]:
    torch = sys.modules["torch"]
    ref = reference.detach()
    cand = candidate.detach()
    diff = (cand - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(ref, args.atol_floor))
    rel = diff / denom
    max_flat_index = int(torch.argmax(diff.reshape(-1)).cpu().item())
    max_index = unravel_index(max_flat_index, diff.shape)
    return {
        "shape": list(diff.shape),
        "max": {
            "shot": int(max_index[0]),
            "time": int(max_index[1]),
            "receiver": int(max_index[2]),
            "abs_diff": index_record(diff, max_index),
            "rel_diff": index_record(rel, max_index),
            "reference": index_record(ref, max_index),
            "candidate": index_record(cand, max_index),
        },
        "first_diff": first_diff_record(diff, rel, ref, cand, threshold=args.threshold),
        "nonzero_count": int(torch.count_nonzero(diff > args.threshold).cpu().item()),
        "per_shot": per_shot_records(diff, rel),
        "time_distribution": time_distribution(diff, top_k=args.top_k),
        "top_differences": topk_records(diff, rel, ref, cand, top_k=args.top_k),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    pair = run_forward_pair(args)
    reports = {
        component: component_report(pair["production"][component], pair["experimental"][component], args)
        for component in ("p", "u", "w")
    }
    return {
        "status": "ok",
        "purpose": "Locate production vs experimental acoustic receiver-output differences",
        "backend": pair["backend"].diagnostics(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "shape": {
            "shots": pair["propagator"].src_n,
            "batch_size": args.batch_size,
            "receivers": pair["propagator"].rcv_n,
            "nt": pair["propagator"].nt,
            "nx": pair["propagator"].nx,
            "nz": pair["propagator"].nz,
            "checkpoint_segments": args.checkpoint_segments,
        },
        "batch": {
            "begin": pair["batch_range"].begin,
            "end": pair["batch_range"].end,
            "shot_index": pair["batch_range"].shot_index.tolist(),
        },
        "timings": pair["timings"],
        "components": reports,
        "summary": {
            component: {
                "max_abs_diff": report["max"]["abs_diff"],
                "max_rel_diff": report["max"]["rel_diff"],
                "first_diff": report["first_diff"],
                "first_nonzero_time": report["time_distribution"]["first_nonzero_time"],
                "nonzero_count": report["nonzero_count"],
            }
            for component, report in reports.items()
        },
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--atol-floor", type=float, default=1e-12)
    parser.add_argument("--top-k", type=int, default=10)
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
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
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
