#!/usr/bin/env python
"""Measure coarse backward stages for the remat pressure candidate.

This diagnostic is lighter than ``torch.autograd.profiler``. It enables the
opt-in stage timer in ``acoustic_custom_kernels`` and runs the same one-iteration
parity gate used by the checkpoint memory matrix.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ADFWI.propagator.acoustic_custom_kernels import (
    clear_remat_backward_stage_timings,
    get_remat_backward_stage_memory,
    get_remat_backward_stage_timings,
    remat_backward_stage_timing,
)
from scripts.benchmark import acoustic_experimental_forward_iteration_parity as parity
from scripts.benchmark import acoustic_fwi_iteration_profile as profile


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_remat_backward_stage_timing_20260603.json"
)


def run(args: argparse.Namespace):
    clear_remat_backward_stage_timings()
    with remat_backward_stage_timing(True):
        variant = parity.run_variant(args, mode=args.candidate_mode)
    stages = get_remat_backward_stage_timings()
    stage_memory = get_remat_backward_stage_memory()
    backward = variant["iteration"]["timings"]["backward"]
    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Coarse rematerialized pressure backward stage timing",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "variant": parity.public_variant(variant),
        "stage_timings": stages,
        "stage_memory": stage_memory,
        "stage_fraction_of_backward": {
            name: value / backward if backward > 0 else None
            for name, value in stages.items()
        },
        "stage_total_seconds": sum(stages.values()),
        "stage_total_fraction_of_backward": sum(stages.values()) / backward if backward > 0 else None,
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    parser = parity.build_parser(argv)
    parser.description = __doc__
    parser.set_defaults(
        result_json=DEFAULT_OUTPUT,
        candidate_mode="experimental-remat-pressure-chunk",
        loss_mode="observed-pressure",
        waveform_normalize=True,
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.candidate_mode != "experimental-remat-pressure-chunk":
        parser.error("stage timing currently expects --candidate-mode experimental-remat-pressure-chunk")
    if args.save_forward_wavefield:
        parser.error("stage timing requires --no-save-forward-wavefield")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(json.dumps({
        "timings": report["variant"]["iteration"]["timings"],
        "stage_timings": report["stage_timings"],
        "stage_memory": report["stage_memory"],
        "stage_fraction_of_backward": report["stage_fraction_of_backward"],
        "stage_total_fraction_of_backward": report["stage_total_fraction_of_backward"],
    }, indent=2, sort_keys=True))
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
