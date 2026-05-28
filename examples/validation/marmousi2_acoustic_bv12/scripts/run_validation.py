#!/usr/bin/env python
"""Dispatch Marmousi2 acoustic validation stages.

Forward modeling and inversion live in separate script modules:

- ``forward_modeling.py``: true-model forward simulation and observed data.
- ``inversion.py``: initial-model FWI using the forward-generated data.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from forward_modeling import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    add_case_arguments,
    json_default,
    run_check,
    run_forward,
    validate_case_args,
)
from inversion import add_inversion_arguments, run_inversion  # noqa: E402


@dataclass(frozen=True)
class StagePlan:
    name: str
    description: str
    output_root: Path
    parameters: Dict[str, Any]


def stage_iterations(args: argparse.Namespace, stage: str) -> int:
    if args.iterations is not None:
        return args.iterations
    if stage == "inversion100":
        return 100
    return 10


def build_stage_plans(args: argparse.Namespace) -> List[StagePlan]:
    stages = ["check", "forward", "inversion10"] if args.stage == "all" else [args.stage]
    plans = []
    for stage in stages:
        parameters = {
            "device": args.device,
            "dtype": args.dtype,
            "shots": args.shots,
            "nt": args.nt,
            "dt": args.dt,
            "f0": args.f0,
            "checkpoint_segments": args.checkpoint_segments,
            "output_root": args.output_root,
        }
        if stage in {"inversion10", "inversion100"}:
            parameters.update({"iterations": stage_iterations(args, stage), "lr": args.lr})
        plans.append(
            StagePlan(
                name=stage,
                description={
                    "check": "build true model, survey, backend, and propagator without propagation",
                    "forward": "forward_modeling.py true-model forward that writes obs_data.npz",
                    "inversion10": "inversion.py FWI using forward-generated obs_data.npz",
                    "inversion100": "inversion.py longer FWI using forward-generated obs_data.npz",
                }[stage],
                output_root=args.output_root,
                parameters=parameters,
            )
        )
    return plans


def dry_run_payload(plans: Iterable[StagePlan]) -> Dict[str, object]:
    return {
        "status": "ok",
        "stages": [
            {
                "stage": plan.name,
                "description": plan.description,
                "output_root": plan.output_root,
                "parameters": plan.parameters,
            }
            for plan in plans
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "forward", "inversion10", "inversion100", "all"))
    add_case_arguments(parser)
    parser.set_defaults(output_root=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    add_inversion_arguments(parser)
    parser.set_defaults(iterations=None)
    return parser


def run_stage(args: argparse.Namespace, stage: str) -> Dict[str, Any]:
    if stage == "check":
        return run_check(args)
    if stage == "forward":
        return run_forward(args)
    if stage in {"inversion10", "inversion100"}:
        return run_inversion(args, iterations=stage_iterations(args, stage))
    raise ValueError(f"unsupported stage: {stage}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_case_args(parser, args)
    if args.iterations is not None and args.iterations <= 0:
        parser.error("--iterations must be positive")

    plans = build_stage_plans(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(plans), indent=2, sort_keys=True, default=json_default))
        return 0

    reports = []
    stages = [plan.name for plan in plans]
    try:
        for stage in stages:
            reports.append(run_stage(args, stage))
    except Exception as exc:
        print(json.dumps({"status": "failed", "stage": stage, "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps({"status": "ok", "stages": reports}, indent=2, sort_keys=True, default=json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
