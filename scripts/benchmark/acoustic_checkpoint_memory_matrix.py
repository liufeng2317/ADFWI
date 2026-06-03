#!/usr/bin/env python
"""Compare acoustic checkpoint memory and speed limits on one FWI iteration.

This benchmark is intentionally matrix-shaped: each variant is identified by a
forward mode and a checkpoint segment count, then compared against a selected
reference variant. It does not edit production kernels.
"""

from __future__ import annotations

import argparse
import json
import sys
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
    / "acoustic_checkpoint_memory_matrix_20260603.json"
)
FORWARD_MODES = {
    "experimental",
    "experimental-chunk",
    "experimental-compressed-chunk",
    "experimental-pressure-divergence-chunk",
    "experimental-velocity-divergence-chunk",
    "experimental-remat-chunk",
    "experimental-remat-pressure-chunk",
    "production",
    "production-custom-chunk",
}


def parse_variant(value: str) -> Dict[str, Any]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("variant must use mode:checkpoint_segments, for example production:10")
    mode, segments = value.rsplit(":", 1)
    mode = mode.strip()
    try:
        checkpoint_segments = int(segments)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid checkpoint segment count in {value!r}") from exc
    if checkpoint_segments < 1:
        raise argparse.ArgumentTypeError("checkpoint segment count must be positive")
    if mode not in FORWARD_MODES:
        raise argparse.ArgumentTypeError(f"unknown forward mode {mode!r}; valid modes: {sorted(FORWARD_MODES)}")
    return {"label": f"{mode}:ckpt{checkpoint_segments}", "mode": mode, "checkpoint_segments": checkpoint_segments}


def parse_variants(value: str) -> list[Dict[str, Any]]:
    variants = [parse_variant(item.strip()) for item in value.split(",") if item.strip()]
    if not variants:
        raise argparse.ArgumentTypeError("at least one variant is required")
    labels = [item["label"] for item in variants]
    if len(labels) != len(set(labels)):
        raise argparse.ArgumentTypeError("variant labels must be unique")
    return variants


def variant_args(args: argparse.Namespace, variant: Dict[str, Any]) -> argparse.Namespace:
    values = vars(args).copy()
    values["checkpoint_segments"] = variant["checkpoint_segments"]
    return argparse.Namespace(**values)


def compare_to_reference(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    comparison = parity.compare(reference, candidate)
    reference_total = reference["iteration"]["timing_total"]
    candidate_total = candidate["iteration"]["timing_total"]
    reference_peak = reference["memory"]["peak_allocated_mib"]
    candidate_peak = candidate["memory"]["peak_allocated_mib"]
    comparison["candidate_vs_reference"] = {
        "total_speedup": reference_total / candidate_total,
        "forward_speedup": reference["iteration"]["timings"]["forward"] / candidate["iteration"]["timings"]["forward"],
        "backward_speedup": reference["iteration"]["timings"]["backward"] / candidate["iteration"]["timings"]["backward"],
        "peak_memory_ratio": candidate_peak / reference_peak if reference_peak else None,
    }
    return comparison


def public_summary(label: str, variant: Dict[str, Any], reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    iteration = variant["iteration"]
    item = {
        "label": label,
        "checkpoint_segments": variant["shape"]["checkpoint_segments"],
        "loss": iteration["loss"],
        "raw_grad_finite": iteration["raw_grad_finite"],
        "timing_seconds": {
            "forward": iteration["timings"]["forward"],
            "backward": iteration["timings"]["backward"],
            "total": iteration["timing_total"],
        },
        "peak_memory_mib": variant["memory"]["peak_allocated_mib"],
    }
    if reference is not None:
        comparison = compare_to_reference(reference, variant)
        item["vs_reference"] = {
            "loss_abs_diff": comparison["loss_abs_diff"],
            "raw_grad_max_abs_diff": comparison["raw_grad"]["max_abs_diff"],
            "raw_grad_max_rel_diff": comparison["raw_grad"]["max_rel_diff"],
            "output_max_abs_diff": max(
                diff["max_abs_diff"]
                for batch in comparison["outputs"]
                for diff in batch.values()
            ),
            "output_max_rel_diff": max(
                diff["max_rel_diff"]
                for batch in comparison["outputs"]
                for diff in batch.values()
            ),
            **comparison["candidate_vs_reference"],
        }
    return item


def run(args: argparse.Namespace) -> Dict[str, Any]:
    variants = args.matrix_variants
    reference_label = args.reference_variant
    labels = {item["label"] for item in variants}
    if reference_label not in labels:
        raise ValueError(f"--reference-variant {reference_label!r} must be present in --matrix-variants")

    raw_runs: Dict[str, Dict[str, Any]] = {}
    for variant in variants:
        run_args = variant_args(args, variant)
        raw_runs[variant["label"]] = parity.run_variant(run_args, mode=variant["mode"])

    reference = raw_runs[reference_label]
    summaries = [
        public_summary(label, raw_runs[label], None if label == reference_label else reference)
        for label in [item["label"] for item in variants]
    ]
    pairs = {
        label: {
            "reference": parity.public_variant(reference),
            "candidate": parity.public_variant(raw_runs[label]),
            "comparison": compare_to_reference(reference, raw_runs[label]),
        }
        for label in raw_runs
        if label != reference_label
    }
    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Acoustic checkpoint=1/segmented/custom chunk speed-memory matrix",
        "reference_variant": reference_label,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"matrix_variants"}
        },
        "variants": summaries,
        "pairs": pairs,
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    parser = parity.build_parser(argv)
    parser.description = __doc__
    parser.set_defaults(result_json=DEFAULT_OUTPUT)
    parser.add_argument(
        "--matrix-variants",
        type=parse_variants,
        default=parse_variants("production:10,production:1,production-custom-chunk:10,experimental-chunk:1"),
        help=(
            "Comma separated mode:checkpoint_segments variants. Default compares "
            "segmented production, checkpoint=1 production, segmented production "
            "custom chunk, and full experimental chunk."
        ),
    )
    parser.add_argument(
        "--reference-variant",
        default="production:ckpt10",
        help="Variant label used as numerical and memory reference.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.save_forward_wavefield:
        parser.error("checkpoint memory matrix requires --no-save-forward-wavefield")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(json.dumps(report["variants"], indent=2, sort_keys=True))
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
