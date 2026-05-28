#!/usr/bin/env python
"""Run staged Marmousi2 acoustic validation for bv1.2.

This wrapper keeps validation outputs separate from the original Marmousi2
example notebooks while reusing the maintained script-style entry points under
``scripts/examples``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[3]
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = THIS_DIR / "outputs"
BACKEND_CHECK_SCRIPT = REPO_ROOT / "scripts" / "examples" / "marmousi2_acoustic_backend_check.py"
INVERSION_SCRIPT = REPO_ROOT / "scripts" / "examples" / "marmousi2_acoustic_reduced_inversion.py"


@dataclass(frozen=True)
class StagePlan:
    name: str
    description: str
    command: List[str]
    output_dir: Path


def safe_device_name(device: str) -> str:
    return device.replace(":", "")


def extract_json(stdout: str) -> Optional[Dict[str, object]]:
    start = stdout.find("{")
    if start < 0:
        return None
    return json.loads(stdout[start:])


def build_check_stage(args: argparse.Namespace, *, run_forward: bool) -> StagePlan:
    name = "forward" if run_forward else "check"
    output_dir = args.output_root / f"{name}_{safe_device_name(args.device)}"
    command = [
        args.python,
        str(BACKEND_CHECK_SCRIPT),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--checkpoint-segments",
        str(args.checkpoint_segments),
    ]
    if args.fallback_cpu:
        command.append("--fallback-cpu")
    if run_forward:
        command.extend(
            [
                "--model-file",
                "true_model.npz",
                "--run-forward",
                "--shot-index",
                str(args.forward_shot_index),
            ]
        )
    return StagePlan(
        name=name,
        description="single-shot true-model forward check" if run_forward else "read-only Marmousi2 case check",
        command=command,
        output_dir=output_dir,
    )


def inversion_iterations(args: argparse.Namespace, stage: str) -> int:
    if args.iterations is not None:
        return args.iterations
    if stage == "inversion100":
        return 100
    return 10


def build_inversion_stage(args: argparse.Namespace, *, stage: str) -> StagePlan:
    iterations = inversion_iterations(args, stage)
    output_dir = args.output_root / (
        f"{stage}_{safe_device_name(args.device)}_shot{args.shots}_iter{iterations}_ckpt{args.checkpoint_segments}"
    )
    command = [
        args.python,
        str(INVERSION_SCRIPT),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--shot-count",
        str(args.shots),
        "--nt-samples",
        str(args.nt_samples),
        "--observed-source",
        "synthetic-true",
        "--iterations",
        str(iterations),
        "--optimizer",
        "adam",
        "--lr",
        str(args.lr),
        "--scheduler-step-size",
        str(args.scheduler_step_size),
        "--scheduler-gamma",
        str(args.scheduler_gamma),
        "--misfit",
        "legacy-l2",
        "--waveform-normalize",
        "--auto-update-rho",
        "--checkpoint-segments",
        str(args.checkpoint_segments),
        "--gradient-processor",
        args.gradient_processor,
        "--output-dir",
        str(output_dir),
    ]
    if args.fallback_cpu:
        command.append("--fallback-cpu")
    return StagePlan(
        name=stage,
        description=f"synthetic-true Marmousi2 inversion, {args.shots} shots, {iterations} iterations",
        command=command,
        output_dir=output_dir,
    )


def build_stage_plans(args: argparse.Namespace) -> List[StagePlan]:
    stages = ["check", "forward", "inversion10"] if args.stage == "all" else [args.stage]
    plans: List[StagePlan] = []
    for stage in stages:
        if stage == "check":
            plans.append(build_check_stage(args, run_forward=False))
        elif stage == "forward":
            plans.append(build_check_stage(args, run_forward=True))
        elif stage in {"inversion10", "inversion100"}:
            plans.append(build_inversion_stage(args, stage=stage))
        else:
            raise ValueError(f"unsupported stage: {stage}")
    return plans


def write_stage_files(plan: StagePlan, *, stdout: str, stderr: str, returncode: int) -> None:
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "command.json").write_text(
        json.dumps(
            {
                "stage": plan.name,
                "description": plan.description,
                "command": plan.command,
                "command_text": " ".join(plan.command),
                "returncode": returncode,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (plan.output_dir / "stdout.txt").write_text(stdout)
    (plan.output_dir / "stderr.txt").write_text(stderr)
    summary = extract_json(stdout)
    if summary is not None:
        (plan.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def run_stage(plan: StagePlan, *, overwrite: bool) -> int:
    if plan.output_dir.exists():
        if not overwrite:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "stage": plan.name,
                        "error": f"output directory exists: {plan.output_dir}; pass --overwrite",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(plan.output_dir)
    plan.output_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(plan.command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    write_stage_files(plan, stdout=proc.stdout, stderr=proc.stderr, returncode=proc.returncode)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc.returncode


def dry_run_payload(plans: Iterable[StagePlan]) -> Dict[str, object]:
    return {
        "status": "ok",
        "stages": [
            {
                "stage": plan.name,
                "description": plan.description,
                "output_dir": str(plan.output_dir),
                "command": plan.command,
                "command_text": " ".join(plan.command),
            }
            for plan in plans
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("check", "forward", "inversion10", "inversion100", "all"))
    parser.add_argument("--device", default="npu:0", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--forward-shot-index", type=int, default=0)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--nt-samples", type=int, default=3000)
    parser.add_argument("--iterations", type=int, help="override inversion iterations")
    parser.add_argument("--checkpoint-segments", type=int, default=10)
    parser.add_argument("--lr", type=float, default=10.0)
    parser.add_argument("--scheduler-step-size", type=int, default=200)
    parser.add_argument("--scheduler-gamma", type=float, default=0.75)
    parser.add_argument("--gradient-processor", choices=("legacy", "torch"), default="legacy")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plans = build_stage_plans(args)
    if args.dry_run:
        print(json.dumps(dry_run_payload(plans), indent=2, sort_keys=True))
        return 0

    for plan in plans:
        returncode = run_stage(plan, overwrite=args.overwrite)
        if returncode != 0:
            return returncode
    print(
        json.dumps(
            {
                "status": "ok",
                "stages": [asdict(plan) | {"output_dir": str(plan.output_dir)} for plan in plans],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
