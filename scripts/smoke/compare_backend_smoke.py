#!/usr/bin/env python
"""Compare mini inversion smoke results across devices.

The tool runs the acoustic or elastic mini inversion smoke script once per
requested device, parses each JSON result, and reports absolute/relative drift
for the core numerical metrics. It writes no notebooks, figures, wavefields, or
example outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SCRIPTS = {
    "acoustic": Path("scripts/smoke/acoustic_mini_inversion_smoke.py"),
    "elastic": Path("scripts/smoke/elastic_mini_inversion_smoke.py"),
}
METRICS = ("loss", "vp_grad_norm", "vp_update_norm")
CASES = ("baseline", "mute-offset", "mute-late", "mute-combined", "legacy-lowpass", "trace-missing")


def parse_csv(value: str, choices: Sequence[str], label: str) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    invalid = [item for item in items if item not in choices]
    if invalid:
        valid = ", ".join(choices)
        raise argparse.ArgumentTypeError(f"unsupported {label}: {', '.join(invalid)}; supported: {valid}")
    return items


def parse_problems(value: str) -> List[str]:
    return parse_csv(value, tuple(SMOKE_SCRIPTS), "problem")


def parse_cases(value: str) -> List[str]:
    return parse_csv(value, CASES, "case")


def parse_devices(value: str) -> List[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if len(devices) < 2:
        raise argparse.ArgumentTypeError("at least two devices are required, e.g. cpu,npu:0")
    return devices


def extract_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("child process did not print a JSON object")
    return json.loads(stdout[start:])


def case_args(problem: str, case: str) -> List[str]:
    if case == "baseline":
        return []
    if case == "mute-offset":
        return ["--mute-offset", "15"]
    if case == "mute-late":
        return ["--nt", "160", "--mute-late-window", "0.01"]
    if case == "mute-combined":
        return ["--nt", "160", "--mute-offset", "15", "--mute-late-window", "0.01"]
    if case == "legacy-lowpass":
        return ["--cutoff-freq", "60", "--lowpass-mode", "legacy"]
    if case == "trace-missing":
        return ["--receiver-mask-mode", "select"]
    raise ValueError(f"unsupported case for {problem}: {case}")


def build_child_command(problem: str, case: str, device: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO_ROOT / SMOKE_SCRIPTS[problem]),
        "--device",
        device,
        "--prefer",
        args.prefer,
        "--dtype",
        args.dtype,
        "--checkpoint-segments",
        str(args.checkpoint_segments),
        "--seed",
        str(args.seed),
    ]
    if problem == "acoustic":
        cmd.extend(["--misfit", args.misfit])
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    cmd.extend(case_args(problem, case))
    cmd.extend(args.extra_args)
    return cmd


def run_one(problem: str, case: str, device: str, args: argparse.Namespace) -> Dict[str, Any]:
    cmd = build_child_command(problem, case, device, args)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    run: Dict[str, Any] = {
        "device_request": device,
        "returncode": proc.returncode,
        "command": cmd,
    }
    if proc.stdout.strip():
        try:
            run["result"] = extract_json(proc.stdout)
        except Exception as exc:
            run["stdout"] = proc.stdout
            run["parse_error"] = repr(exc)
    if proc.stderr.strip() and (args.include_stderr or proc.returncode != 0):
        run["stderr"] = proc.stderr

    if proc.returncode == 2 and args.skip_unavailable:
        run["status"] = "unavailable"
    elif proc.returncode != 0:
        run["status"] = "failed"
    else:
        run["status"] = "ok"
    return run


def metric_value(run: Mapping[str, Any], metric: str) -> float:
    result = run.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"run has no parsed result: {run.get('device_request')}")
    inversion = result.get("inversion")
    if not isinstance(inversion, Mapping) or metric not in inversion:
        raise ValueError(f"metric {metric!r} missing for {run.get('device_request')}")
    value = float(inversion[metric])
    if not math.isfinite(value):
        raise ValueError(f"metric {metric!r} is not finite for {run.get('device_request')}: {value}")
    return value


def compare_value(reference: float, candidate: float, rel_tol: float, abs_tol: float) -> Dict[str, Any]:
    abs_diff = abs(candidate - reference)
    denom = max(abs(reference), abs(candidate), 1e-30)
    rel_diff = abs_diff / denom
    passed = abs_diff <= abs_tol or rel_diff <= rel_tol
    return {
        "reference": reference,
        "candidate": candidate,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "abs_tol": abs_tol,
        "rel_tol": rel_tol,
        "passed": passed,
    }


def compare_runs(runs: Sequence[Mapping[str, Any]], metrics: Iterable[str], rel_tol: float, abs_tol: float) -> Dict[str, Any]:
    ok_runs = [run for run in runs if run.get("status") == "ok"]
    if len(ok_runs) < 2:
        return {"status": "failed", "error": "at least two successful runs are required for comparison"}

    reference = ok_runs[0]
    comparisons: Dict[str, Any] = {}
    failed = False
    for candidate in ok_runs[1:]:
        device_key = str(candidate.get("device_request"))
        metric_results: Dict[str, Any] = {}
        for metric in metrics:
            result = compare_value(
                metric_value(reference, metric),
                metric_value(candidate, metric),
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            metric_results[metric] = result
            failed = failed or not result["passed"]
        comparisons[device_key] = metric_results

    return {
        "status": "failed" if failed else "ok",
        "reference_device": reference.get("device_request"),
        "comparisons": comparisons,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare mini inversion smoke results across devices.")
    parser.add_argument("--problem", choices=sorted(SMOKE_SCRIPTS), default="acoustic", help="single problem shortcut")
    parser.add_argument("--case", choices=CASES, default="baseline", help="single case shortcut")
    parser.add_argument("--problems", type=parse_problems, default=None, help="comma-separated problems for matrix mode")
    parser.add_argument("--cases", type=parse_cases, default=None, help="comma-separated cases for matrix mode")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("cpu,npu:0"))
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--strict-unavailable", action="store_false", dest="skip_unavailable", help="fail if a requested device is unavailable")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--misfit", default="L2", help="acoustic mini inversion misfit")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--rel-tol", type=float, default=1e-5)
    parser.add_argument("--abs-tol", type=float, default=1e-10)
    parser.add_argument("--include-stderr", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="extra arguments passed to the underlying smoke script after --")
    parser.set_defaults(skip_unavailable=True)
    return parser


def run_comparison(problem: str, case: str, args: argparse.Namespace) -> Dict[str, Any]:
    runs = [run_one(problem, case, device, args) for device in args.devices]
    failed_runs = [run for run in runs if run.get("status") == "failed"]
    comparison = compare_runs(runs, METRICS, rel_tol=args.rel_tol, abs_tol=args.abs_tol)
    status = "failed" if failed_runs or comparison.get("status") == "failed" else "ok"
    return {
        "status": status,
        "problem": problem,
        "case": case,
        "devices": args.devices,
        "metrics": list(METRICS),
        "runs": runs,
        "comparison": comparison,
    }


def selected_matrix(args: argparse.Namespace) -> tuple[List[str], List[str]]:
    problems = args.problems if args.problems is not None else [args.problem]
    cases = args.cases if args.cases is not None else [args.case]
    return problems, cases


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]

    problems, cases = selected_matrix(args)
    reports = [run_comparison(problem, case, args) for problem in problems for case in cases]
    status = "failed" if any(report.get("status") == "failed" for report in reports) else "ok"

    if len(reports) == 1 and args.problems is None and args.cases is None:
        report = reports[0]
    else:
        report = {
            "status": status,
            "problems": problems,
            "cases": cases,
            "devices": args.devices,
            "metrics": list(METRICS),
            "reports": reports,
        }

    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if status == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
