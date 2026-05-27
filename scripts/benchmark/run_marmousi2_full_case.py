#!/usr/bin/env python
"""Run fixed Marmousi2 full-case benchmark presets.

The presets mirror the current bv1.2 full-case NPU baselines so repeated runs do
not require copying long environment-variable commands.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
INVERSION_SCRIPT = REPO_ROOT / "scripts" / "examples" / "marmousi2_acoustic_reduced_inversion.py"
COMPARE_SCRIPT = REPO_ROOT / "scripts" / "benchmark" / "compare_full_case_outputs.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "tests" / "full_cases" / "outputs"


@dataclass(frozen=True)
class Marmousi2Preset:
    name: str
    description: str
    output_name: str
    shot_count: int
    iterations: int
    checkpoint_segments: int
    nt_samples: int = 3000
    lr: float = 10.0
    scheduler_step_size: int = 200
    scheduler_gamma: float = 0.75


PRESETS: Dict[str, Marmousi2Preset] = {
    "shot3": Marmousi2Preset(
        name="shot3",
        description="Fastest current full-case per-iteration NPU baseline",
        output_name="marmousi2_npu_shot3_ckpt10_iter10",
        shot_count=3,
        iterations=10,
        checkpoint_segments=10,
    ),
    "shot5": Marmousi2Preset(
        name="shot5",
        description="Current full-case throughput stress baseline",
        output_name="marmousi2_npu_shot5_ckpt10_iter10",
        shot_count=5,
        iterations=10,
        checkpoint_segments=10,
    ),
}


def default_output_dir(preset: Marmousi2Preset, device: str) -> Path:
    device_name = device.replace(":", "")
    if device == "npu:0":
        return DEFAULT_OUTPUT_ROOT / preset.output_name
    return DEFAULT_OUTPUT_ROOT / f"{preset.output_name}_{device_name}"


def build_command(
    preset: Marmousi2Preset,
    *,
    device: str,
    output_dir: Path,
    python: str,
    gradient_processor: str = "legacy",
) -> List[str]:
    return [
        python,
        str(INVERSION_SCRIPT),
        "--device",
        device,
        "--shot-count",
        str(preset.shot_count),
        "--nt-samples",
        str(preset.nt_samples),
        "--observed-source",
        "synthetic-true",
        "--iterations",
        str(preset.iterations),
        "--optimizer",
        "adam",
        "--lr",
        str(preset.lr),
        "--scheduler-step-size",
        str(preset.scheduler_step_size),
        "--scheduler-gamma",
        str(preset.scheduler_gamma),
        "--misfit",
        "legacy-l2",
        "--waveform-normalize",
        "--auto-update-rho",
        "--checkpoint-segments",
        str(preset.checkpoint_segments),
        "--gradient-processor",
        gradient_processor,
        "--output-dir",
        str(output_dir),
    ]


def build_compare_command(args: argparse.Namespace, *, output_dir: Path) -> Optional[List[str]]:
    if args.compare_to is None:
        return None
    command = [
        args.python,
        str(COMPARE_SCRIPT),
        str(args.compare_to),
        str(output_dir),
        "--labels",
        args.compare_labels,
    ]
    if args.fail_on_loss_drift:
        command.extend(
            [
                "--fail-on-loss-drift",
                "--loss-abs-tol",
                str(args.loss_abs_tol),
                "--loss-rel-tol",
                str(args.loss_rel_tol),
            ]
        )
    return command


def profile_output_path(args: argparse.Namespace, *, output_dir: Path) -> Optional[Path]:
    if not args.profile:
        return None
    if args.profile_output is not None:
        return args.profile_output
    return output_dir / "python_profile.prof"


def build_profiled_command(command: List[str], *, python: str, profile_output: Optional[Path]) -> List[str]:
    if profile_output is None:
        return command
    return [python, "-m", "cProfile", "-o", str(profile_output), *command[1:]]


def build_plan(args: argparse.Namespace) -> Dict[str, object]:
    preset = PRESETS[args.preset]
    output_dir = args.output_dir if args.output_dir is not None else default_output_dir(preset, args.device)
    base_command = build_command(
        preset,
        device=args.device,
        output_dir=output_dir,
        python=args.python,
        gradient_processor=args.gradient_processor,
    )
    profile_output = profile_output_path(args, output_dir=output_dir)
    command = build_profiled_command(base_command, python=args.python, profile_output=profile_output)
    compare_command = build_compare_command(args, output_dir=output_dir)
    return {
        "status": "ok",
        "preset": asdict(preset),
        "device": args.device,
        "output_dir": str(output_dir),
        "overwrite": args.overwrite,
        "profile": args.profile,
        "profile_output": None if profile_output is None else str(profile_output),
        "gradient_processor": args.gradient_processor,
        "base_command": base_command,
        "base_command_text": " ".join(base_command),
        "command": command,
        "command_text": " ".join(command),
        "compare_command": compare_command,
        "compare_command_text": None if compare_command is None else " ".join(compare_command),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", choices=sorted(PRESETS), help="Full-case preset to run")
    parser.add_argument("--device", default="npu:0", help="Backend device request")
    parser.add_argument("--output-dir", type=Path, help="Override output directory")
    parser.add_argument("--python", default=sys.executable, help="Python executable for the inversion script")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing output directory before running")
    parser.add_argument("--dry-run", action="store_true", help="Print the preset plan without running the inversion")
    parser.add_argument("--list-presets", action="store_true", help="Print available presets and exit")
    parser.add_argument("--gradient-processor", choices=("legacy", "torch"), default="legacy", help="Gradient processor implementation passed to the inversion script")
    parser.add_argument("--profile", action="store_true", help="Run the preset under Python cProfile")
    parser.add_argument("--profile-output", type=Path, help="cProfile output path; defaults to output_dir/python_profile.prof")
    parser.add_argument("--compare-to", type=Path, help="Optional baseline output directory or summary.json to compare after the run")
    parser.add_argument("--compare-labels", default="baseline,candidate", help="Labels passed to compare_full_case_outputs.py")
    parser.add_argument("--fail-on-loss-drift", action="store_true", help="Fail when the post-run comparison exceeds loss tolerances")
    parser.add_argument("--loss-abs-tol", type=float, default=1e-5)
    parser.add_argument("--loss-rel-tol", type=float, default=1e-8)
    return parser.parse_args(argv)


def list_presets() -> Dict[str, object]:
    return {
        "status": "ok",
        "presets": {name: asdict(preset) for name, preset in PRESETS.items()},
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.list_presets:
        print(json.dumps(list_presets(), indent=2, sort_keys=True))
        return 0

    plan = build_plan(args)
    output_dir = Path(plan["output_dir"])
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if output_dir.exists():
        if not args.overwrite:
            print(
                json.dumps(
                    {
                        "status": "failed",
                        "error": f"output directory exists: {output_dir}; pass --overwrite or choose --output-dir",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
        shutil.rmtree(output_dir)

    proc = subprocess.run(plan["command"], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode != 0:
        return proc.returncode

    compare_command = plan["compare_command"]
    if compare_command is None:
        return 0
    compare_proc = subprocess.run(compare_command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if compare_proc.stdout:
        print(compare_proc.stdout, end="")
    if compare_proc.stderr:
        print(compare_proc.stderr, end="", file=sys.stderr)
    return compare_proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
