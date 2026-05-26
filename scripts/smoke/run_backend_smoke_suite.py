#!/usr/bin/env python
"""Run layered backend smoke checks for ADFWI.

The suite is intended for new CPU/NPU/CUDA machines. It starts with the
lightweight public backend API check and can optionally run tensor-level misfit
checks, acoustic/elastic forward checks, user-facing minimal examples, and
mini-inversion CPU-vs-device comparisons. It writes no notebooks, figures,
wavefields, or example outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLE_DIR = REPO_ROOT / "scripts" / "examples"

SUITES = ("public", "misfit", "acoustic-forward", "elastic-forward", "examples", "compare-mini")
SCRIPT_BY_SUITE = {
    "public": "backend_public_api_smoke.py",
    "misfit": "misfit_backend_smoke.py",
    "acoustic-forward": "acoustic_backend_smoke.py",
    "elastic-forward": "elastic_backend_smoke.py",
}
EXAMPLE_SCRIPT_BY_PROBLEM = {
    "acoustic": "minimal_acoustic_fwi_backend.py",
    "elastic": "minimal_elastic_fwi_backend.py",
}


def parse_csv(value: str, *, label: str, choices: Sequence[str] | None = None) -> List[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError(f"at least one {label} is required")
    if choices is not None:
        invalid = [item for item in items if item not in choices]
        if invalid:
            raise argparse.ArgumentTypeError(f"unsupported {label}: {', '.join(invalid)}; supported: {', '.join(choices)}")
    return items


def parse_devices(value: str) -> List[str]:
    return parse_csv(value, label="device")


def parse_suites(value: str) -> List[str]:
    return parse_csv(value, label="suite", choices=SUITES)


def extract_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("child process did not print a JSON object")
    return json.loads(stdout[start:])


def run_command(cmd: List[str], *, include_stderr: bool) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True)
    run: Dict[str, Any] = {
        "command": cmd,
        "returncode": proc.returncode,
    }
    if proc.stdout.strip():
        try:
            run["result"] = extract_json(proc.stdout)
        except Exception as exc:
            run["stdout"] = proc.stdout
            run["parse_error"] = repr(exc)
    if proc.stderr.strip() and (include_stderr or proc.returncode != 0):
        run["stderr"] = proc.stderr

    if proc.returncode == 2:
        run["status"] = "unavailable"
    elif proc.returncode != 0:
        run["status"] = "failed"
    else:
        run["status"] = "ok"
    return run


def command_for_single_device_suite(suite: str, device: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / SCRIPT_BY_SUITE[suite]),
        "--device",
        device,
        "--prefer",
        args.prefer,
        "--dtype",
        args.dtype,
    ]
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    if suite in {"acoustic-forward", "elastic-forward"}:
        cmd.extend(["--checkpoint-segments", str(args.checkpoint_segments), "--seed", str(args.seed)])
        if args.skip_backward:
            cmd.append("--skip-backward")
    if suite == "misfit":
        cmd.extend(["--misfits", args.misfits])
    return cmd


def run_single_device_suite(suite: str, args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for device in args.devices:
        run = run_command(command_for_single_device_suite(suite, device, args), include_stderr=args.include_stderr)
        run["device_request"] = device
        runs.append(run)
    failed = [run for run in runs if run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)]
    return {
        "suite": suite,
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "runs": runs,
    }


def command_for_example(problem: str, device: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(EXAMPLE_DIR / EXAMPLE_SCRIPT_BY_PROBLEM[problem]),
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
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    return cmd


def example_metric(run: Dict[str, Any], metric: str) -> float | None:
    try:
        return float(run["result"]["inversion"][metric])
    except (KeyError, TypeError, ValueError):
        return None


def compare_example_runs(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    comparisons = []
    metrics = ("loss", "vp_grad_norm", "vp_update_norm")
    for problem in args.example_problems:
        problem_runs = [run for run in runs if run.get("problem") == problem and run["status"] == "ok"]
        reference = next((run for run in problem_runs if run.get("device_request") == "cpu"), None)
        if reference is None and problem_runs:
            reference = problem_runs[0]
        if reference is None:
            continue
        reference_device = reference.get("device_request")
        for run in problem_runs:
            if run is reference:
                continue
            metric_reports = []
            failed = False
            for metric in metrics:
                ref_value = example_metric(reference, metric)
                value = example_metric(run, metric)
                if ref_value is None or value is None:
                    failed = True
                    metric_reports.append({"metric": metric, "status": "missing"})
                    continue
                abs_diff = abs(value - ref_value)
                rel_diff = abs_diff / max(abs(ref_value), args.example_atol)
                metric_failed = abs_diff > args.example_atol and rel_diff > args.example_rtol
                failed = failed or metric_failed
                metric_reports.append({
                    "metric": metric,
                    "reference": ref_value,
                    "value": value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "status": "failed" if metric_failed else "ok",
                })
            comparisons.append({
                "problem": problem,
                "reference_device": reference_device,
                "device": run.get("device_request"),
                "rtol": args.example_rtol,
                "atol": args.example_atol,
                "status": "failed" if failed else "ok",
                "metrics": metric_reports,
            })
    return comparisons


def run_examples(args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for problem in args.example_problems:
        for device in args.devices:
            run = run_command(command_for_example(problem, device, args), include_stderr=args.include_stderr)
            run["problem"] = problem
            run["device_request"] = device
            runs.append(run)
    comparisons = compare_example_runs(runs, args)
    failed = [run for run in runs if run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)]
    failed.extend(compare for compare in comparisons if compare["status"] == "failed")
    return {
        "suite": "examples",
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "problems": args.example_problems,
        "runs": runs,
        "comparisons": comparisons,
    }


def run_compare_mini(args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "compare_backend_smoke.py"),
        "--problems",
        args.compare_problems,
        "--cases",
        args.compare_cases,
        "--devices",
        ",".join(args.devices),
        "--prefer",
        args.prefer,
        "--dtype",
        args.dtype,
        "--checkpoint-segments",
        str(args.checkpoint_segments),
        "--seed",
        str(args.seed),
    ]
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    if args.include_stderr:
        cmd.append("--include-stderr")
    run = run_command(cmd, include_stderr=args.include_stderr)
    failed = run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)
    return {
        "suite": "compare-mini",
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "run": run,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", type=parse_suites, default=parse_suites("public"), help=f"comma-separated suites: {', '.join(SUITES)}")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("cpu,npu:0"), help="comma-separated devices, e.g. cpu,npu:0 or cpu,auto")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--skip-unavailable", action="store_true", default=True, help="treat unavailable requested devices as skipped")
    parser.add_argument("--strict-unavailable", action="store_false", dest="skip_unavailable", help="fail if a requested device is unavailable")
    parser.add_argument("--include-stderr", action="store_true")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--skip-backward", action="store_true", help="skip backward pass for forward propagation suites")
    parser.add_argument("--misfits", default="L2", help="misfit list passed to misfit_backend_smoke.py")
    parser.add_argument("--compare-problems", default="acoustic,elastic", help="problem list passed to compare_backend_smoke.py")
    parser.add_argument("--compare-cases", default="trace-missing", help="case list passed to compare_backend_smoke.py")
    parser.add_argument(
        "--example-problems",
        type=lambda value: parse_csv(value, label="example problem", choices=tuple(EXAMPLE_SCRIPT_BY_PROBLEM)),
        default=parse_csv("acoustic,elastic", label="example problem", choices=tuple(EXAMPLE_SCRIPT_BY_PROBLEM)),
        help="comma-separated user-facing examples to run when --suites includes examples",
    )
    parser.add_argument("--example-rtol", type=float, default=1e-4, help="relative tolerance for CPU-vs-device example metrics")
    parser.add_argument("--example-atol", type=float, default=1e-8, help="absolute tolerance for CPU-vs-device example metrics")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    reports = []
    for suite in args.suites:
        if suite == "compare-mini":
            reports.append(run_compare_mini(args))
        elif suite == "examples":
            reports.append(run_examples(args))
        else:
            reports.append(run_single_device_suite(suite, args))

    failed = [report for report in reports if report["status"] == "failed"]
    report = {
        "status": "failed" if failed else "ok",
        "suites": args.suites,
        "devices": args.devices,
        "prefer": args.prefer,
        "dtype": args.dtype,
        "reports": reports,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
