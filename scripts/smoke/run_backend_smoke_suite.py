#!/usr/bin/env python
"""Run layered backend smoke checks for ADFWI.

The suite is intended for new CPU/NPU/CUDA machines. It starts with the
lightweight public backend API check and can optionally run tensor-level misfit
checks, acoustic/elastic forward checks, user-facing minimal examples,
read-only real-case checks, reduced real-case inversion checks, and mini-inversion CPU-vs-device comparisons.
It writes no notebooks, figures, wavefields, or example outputs.
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

SUITES = ("public", "misfit", "acoustic-forward", "elastic-forward", "examples", "case-checks", "case-inversion", "compare-mini")
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
CASE_CHECK_SCRIPT_BY_CASE = {
    "marmousi2-acoustic": "marmousi2_acoustic_backend_check.py",
}
CASE_INVERSION_SCRIPT_BY_CASE = {
    "marmousi2-acoustic-reduced": "marmousi2_acoustic_reduced_inversion.py",
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


def parse_example_gradient_processors(value: str) -> List[str]:
    return parse_csv(value, label="example gradient processor", choices=("legacy", "torch"))


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


def command_for_example(problem: str, device: str, args: argparse.Namespace, gradient_processor: str | None = None) -> List[str]:
    processor = gradient_processor if gradient_processor is not None else args.example_gradient_processor
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
        "--gradient-processor",
        processor,
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
        for processor in args.example_gradient_processors:
            problem_runs = [
                run
                for run in runs
                if run.get("problem") == problem
                and run.get("gradient_processor") == processor
                and run["status"] == "ok"
            ]
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
                    "comparison_type": "device",
                    "problem": problem,
                    "gradient_processor": processor,
                    "reference_device": reference_device,
                    "device": run.get("device_request"),
                    "rtol": args.example_rtol,
                    "atol": args.example_atol,
                    "status": "failed" if failed else "ok",
                    "metrics": metric_reports,
                })
    return comparisons


def compare_example_gradient_processors(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    comparisons = []
    if len(args.example_gradient_processors) < 2 or "legacy" not in args.example_gradient_processors:
        return comparisons
    metrics = ("loss", "vp_grad_norm", "vp_update_norm")
    for problem in args.example_problems:
        for device in args.devices:
            device_runs = [
                run
                for run in runs
                if run.get("problem") == problem
                and run.get("device_request") == device
                and run["status"] == "ok"
            ]
            reference = next((run for run in device_runs if run.get("gradient_processor") == "legacy"), None)
            if reference is None:
                continue
            for processor in args.example_gradient_processors:
                if processor == "legacy":
                    continue
                run = next((item for item in device_runs if item.get("gradient_processor") == processor), None)
                if run is None:
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
                    rel_diff = abs_diff / max(abs(ref_value), args.example_gradient_atol)
                    metric_failed = abs_diff > args.example_gradient_atol and rel_diff > args.example_gradient_rtol
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
                    "comparison_type": "gradient_processor",
                    "problem": problem,
                    "device": device,
                    "reference_gradient_processor": "legacy",
                    "gradient_processor": processor,
                    "rtol": args.example_gradient_rtol,
                    "atol": args.example_gradient_atol,
                    "status": "failed" if failed else "ok",
                    "metrics": metric_reports,
                })
    return comparisons


def run_examples(args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for problem in args.example_problems:
        for device in args.devices:
            for processor in args.example_gradient_processors:
                run = run_command(command_for_example(problem, device, args, processor), include_stderr=args.include_stderr)
                run["problem"] = problem
                run["device_request"] = device
                run["gradient_processor"] = processor
                runs.append(run)
    comparisons = compare_example_runs(runs, args)
    comparisons.extend(compare_example_gradient_processors(runs, args))
    failed = [run for run in runs if run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)]
    failed.extend(compare for compare in comparisons if compare["status"] == "failed")
    return {
        "suite": "examples",
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "problems": args.example_problems,
        "gradient_processors": args.example_gradient_processors,
        "runs": runs,
        "comparisons": comparisons,
    }


def command_for_case_check(case: str, device: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(EXAMPLE_DIR / CASE_CHECK_SCRIPT_BY_CASE[case]),
        "--device",
        device,
        "--prefer",
        args.prefer,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--model-file",
        args.case_model_file,
        "--checkpoint-segments",
        str(args.checkpoint_segments),
    ]
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    if args.case_run_forward:
        cmd.extend(["--run-forward", "--shot-index", str(args.case_shot_index)])
    return cmd


def case_forward_norm(run: Dict[str, Any]) -> float | None:
    try:
        forward = run["result"]["single_shot_forward"]
        if forward is None:
            return None
        return float(forward["pressure"]["norm"])
    except (KeyError, TypeError, ValueError):
        return None


def compare_case_forward_runs(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    comparisons = []
    if not args.case_run_forward:
        return comparisons
    for case in args.case_checks:
        case_runs = [run for run in runs if run.get("case_check") == case and run["status"] == "ok"]
        reference = next((run for run in case_runs if run.get("device_request") == "cpu"), None)
        if reference is None and case_runs:
            reference = case_runs[0]
        if reference is None:
            continue
        reference_norm = case_forward_norm(reference)
        for run in case_runs:
            if run is reference:
                continue
            value = case_forward_norm(run)
            failed = reference_norm is None or value is None
            metric: Dict[str, Any] = {"metric": "single_shot_forward.pressure.norm"}
            if failed:
                metric["status"] = "missing"
            else:
                abs_diff = abs(value - reference_norm)
                rel_diff = abs_diff / max(abs(reference_norm), args.case_forward_atol)
                failed = abs_diff > args.case_forward_atol and rel_diff > args.case_forward_rtol
                metric.update({
                    "reference": reference_norm,
                    "value": value,
                    "abs_diff": abs_diff,
                    "rel_diff": rel_diff,
                    "status": "failed" if failed else "ok",
                })
            comparisons.append({
                "case_check": case,
                "reference_device": reference.get("device_request"),
                "device": run.get("device_request"),
                "rtol": args.case_forward_rtol,
                "atol": args.case_forward_atol,
                "status": "failed" if failed else "ok",
                "metrics": [metric],
            })
    return comparisons


def run_case_checks(args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for case in args.case_checks:
        for device in args.devices:
            run = run_command(command_for_case_check(case, device, args), include_stderr=args.include_stderr)
            run["case_check"] = case
            run["device_request"] = device
            runs.append(run)
    comparisons = compare_case_forward_runs(runs, args)
    failed = [run for run in runs if run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)]
    failed.extend(compare for compare in comparisons if compare["status"] == "failed")
    return {
        "suite": "case-checks",
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "case_checks": args.case_checks,
        "runs": runs,
        "comparisons": comparisons,
    }


def command_for_case_inversion(case: str, device: str, args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        str(EXAMPLE_DIR / CASE_INVERSION_SCRIPT_BY_CASE[case]),
        "--device",
        device,
        "--prefer",
        args.prefer,
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--model-file",
        args.case_model_file,
        "--checkpoint-segments",
        str(args.checkpoint_segments),
        "--shot-count",
        str(args.case_inversion_shot_count),
        "--nt-samples",
        str(args.case_inversion_nt_samples),
        "--iterations",
        str(args.case_inversion_iterations),
        "--optimizer",
        args.case_inversion_optimizer,
        "--misfit",
        args.case_inversion_misfit,
        "--lr",
        str(args.case_inversion_lr),
        "--scheduler-step-size",
        str(args.case_inversion_scheduler_step_size),
        "--scheduler-gamma",
        str(args.case_inversion_scheduler_gamma),
        "--dt-for-loss",
        str(args.case_inversion_dt_for_loss),
        "--grad-mute-top",
        str(args.case_inversion_grad_mute_top),
    ]
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    if args.case_inversion_norm_grad:
        cmd.append("--norm-grad")
    if args.case_inversion_forw_illumination:
        cmd.append("--forw-illumination")
    if args.case_inversion_auto_update_rho:
        cmd.append("--auto-update-rho")
    if args.case_inversion_waveform_normalize:
        cmd.append("--waveform-normalize")
    return cmd


def inversion_metric(run: Dict[str, Any], metric: str) -> float | None:
    try:
        return float(run["result"]["inversion"][metric])
    except (KeyError, TypeError, ValueError):
        return None


def compare_case_inversion_runs(runs: List[Dict[str, Any]], args: argparse.Namespace) -> List[Dict[str, Any]]:
    comparisons = []
    metrics = ("loss", "vp_grad_norm", "vp_update_norm")
    for case in args.case_inversions:
        case_runs = [run for run in runs if run.get("case_inversion") == case and run["status"] == "ok"]
        reference = next((run for run in case_runs if run.get("device_request") == "cpu"), None)
        if reference is None and case_runs:
            reference = case_runs[0]
        if reference is None:
            continue
        for run in case_runs:
            if run is reference:
                continue
            metric_reports = []
            failed = False
            for metric in metrics:
                ref_value = inversion_metric(reference, metric)
                value = inversion_metric(run, metric)
                if ref_value is None or value is None:
                    failed = True
                    metric_reports.append({"metric": metric, "status": "missing"})
                    continue
                abs_diff = abs(value - ref_value)
                rel_diff = abs_diff / max(abs(ref_value), args.case_inversion_atol)
                metric_failed = abs_diff > args.case_inversion_atol and rel_diff > args.case_inversion_rtol
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
                "case_inversion": case,
                "reference_device": reference.get("device_request"),
                "device": run.get("device_request"),
                "rtol": args.case_inversion_rtol,
                "atol": args.case_inversion_atol,
                "status": "failed" if failed else "ok",
                "metrics": metric_reports,
            })
    return comparisons


def run_case_inversions(args: argparse.Namespace) -> Dict[str, Any]:
    runs = []
    for case in args.case_inversions:
        for device in args.devices:
            run = run_command(command_for_case_inversion(case, device, args), include_stderr=args.include_stderr)
            run["case_inversion"] = case
            run["device_request"] = device
            runs.append(run)
    comparisons = compare_case_inversion_runs(runs, args)
    failed = [run for run in runs if run["status"] == "failed" or (run["status"] == "unavailable" and not args.skip_unavailable)]
    failed.extend(compare for compare in comparisons if compare["status"] == "failed")
    return {
        "suite": "case-inversion",
        "status": "failed" if failed else "ok",
        "devices": args.devices,
        "case_inversions": args.case_inversions,
        "runs": runs,
        "comparisons": comparisons,
    }


def iter_report_runs(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "runs" in report:
        return list(report["runs"])
    if "run" in report:
        return [report["run"]]
    return []


def summarize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    runs = iter_report_runs(report)
    counts = {"ok": 0, "failed": 0, "unavailable": 0}
    for run in runs:
        status = run.get("status", "failed")
        counts[status if status in counts else "failed"] += 1

    summary: Dict[str, Any] = {
        "suite": report["suite"],
        "status": report["status"],
        "runs": len(runs),
        "ok": counts["ok"],
        "failed": counts["failed"],
        "unavailable": counts["unavailable"],
    }
    if report["suite"] in {"examples", "case-checks", "case-inversion"}:
        comparisons = report.get("comparisons", [])
        metric_rows = [metric for comparison in comparisons for metric in comparison.get("metrics", [])]
        summary["comparisons"] = len(comparisons)
        summary["max_abs_diff"] = max((float(metric.get("abs_diff", 0.0)) for metric in metric_rows), default=0.0)
        summary["max_rel_diff"] = max((float(metric.get("rel_diff", 0.0)) for metric in metric_rows), default=0.0)
        summary["comparison_status"] = "failed" if any(comparison.get("status") == "failed" for comparison in comparisons) else "ok"
    return summary


def summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    suite_summaries = [summarize_report(report) for report in reports]
    return {
        "suites": len(suite_summaries),
        "failed_suites": sum(1 for item in suite_summaries if item["status"] == "failed"),
        "runs": sum(item["runs"] for item in suite_summaries),
        "ok": sum(item["ok"] for item in suite_summaries),
        "failed": sum(item["failed"] for item in suite_summaries),
        "unavailable": sum(item["unavailable"] for item in suite_summaries),
        "by_suite": suite_summaries,
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
    parser.add_argument("--misfits", default="L2,SquaredL2", help="misfit list passed to misfit_backend_smoke.py")
    parser.add_argument("--compare-problems", default="acoustic,elastic", help="problem list passed to compare_backend_smoke.py")
    parser.add_argument("--compare-cases", default="trace-missing", help="case list passed to compare_backend_smoke.py")
    parser.add_argument(
        "--example-problems",
        type=lambda value: parse_csv(value, label="example problem", choices=tuple(EXAMPLE_SCRIPT_BY_PROBLEM)),
        default=parse_csv("acoustic,elastic", label="example problem", choices=tuple(EXAMPLE_SCRIPT_BY_PROBLEM)),
        help="comma-separated user-facing examples to run when --suites includes examples",
    )
    parser.add_argument("--example-gradient-processor", choices=("legacy", "torch"), default="legacy", help="gradient processor implementation passed to minimal example scripts")
    parser.add_argument("--example-gradient-processors", type=parse_example_gradient_processors, default=None, help="comma-separated gradient processors for examples; use legacy,torch to compare both paths")
    parser.add_argument("--example-rtol", type=float, default=1e-4, help="relative tolerance for CPU-vs-device example metrics")
    parser.add_argument("--example-atol", type=float, default=1e-8, help="absolute tolerance for CPU-vs-device example metrics")
    parser.add_argument("--example-gradient-rtol", type=float, default=1e-6, help="relative tolerance for legacy-vs-torch example gradient processor metrics")
    parser.add_argument("--example-gradient-atol", type=float, default=1e-12, help="absolute tolerance for legacy-vs-torch example gradient processor metrics")
    parser.add_argument(
        "--case-checks",
        type=lambda value: parse_csv(value, label="case check", choices=tuple(CASE_CHECK_SCRIPT_BY_CASE)),
        default=parse_csv("marmousi2-acoustic", label="case check", choices=tuple(CASE_CHECK_SCRIPT_BY_CASE)),
        help="comma-separated read-only real-case checks to run when --suites includes case-checks",
    )
    parser.add_argument("--case-model-file", default="init_model.npz", choices=("init_model.npz", "true_model.npz"))
    parser.add_argument("--case-run-forward", action="store_true", help="run optional forward checks inside real-case check scripts")
    parser.add_argument("--case-shot-index", type=int, default=0, help="shot index used by optional real-case forward checks")
    parser.add_argument("--case-forward-rtol", type=float, default=1e-4, help="relative tolerance for CPU-vs-device real-case forward metrics")
    parser.add_argument("--case-forward-atol", type=float, default=1e-6, help="absolute tolerance for CPU-vs-device real-case forward metrics")
    parser.add_argument(
        "--case-inversions",
        type=lambda value: parse_csv(value, label="case inversion", choices=tuple(CASE_INVERSION_SCRIPT_BY_CASE)),
        default=parse_csv("marmousi2-acoustic-reduced", label="case inversion", choices=tuple(CASE_INVERSION_SCRIPT_BY_CASE)),
        help="comma-separated reduced real-case inversions to run when --suites includes case-inversion",
    )
    parser.add_argument("--case-inversion-shot-count", type=int, default=1, help="shot count used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-nt-samples", type=int, default=300, help="time samples used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-iterations", type=int, default=1, help="iteration count used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-optimizer", default="sgd", choices=("sgd", "adam"), help="optimizer used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-misfit", default="safe-squared-l2", choices=("safe-squared-l2", "legacy-l2"), help="misfit used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-lr", type=float, default=1e12, help="optimizer learning rate used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-scheduler-step-size", type=int, default=1, help="scheduler step size used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-scheduler-gamma", type=float, default=1.0, help="scheduler gamma used by reduced real-case inversion checks")
    parser.add_argument("--case-inversion-dt-for-loss", type=float, default=1.0, help="dt/weight passed to the reduced inversion loss")
    parser.add_argument("--case-inversion-grad-mute-top", type=int, default=12, help="number of top model rows muted in reduced inversion gradient")
    parser.add_argument("--case-inversion-rtol", type=float, default=1e-4, help="relative tolerance for CPU-vs-device reduced inversion metrics")
    parser.add_argument("--case-inversion-atol", type=float, default=1e-8, help="absolute tolerance for CPU-vs-device reduced inversion metrics")
    parser.add_argument("--case-inversion-norm-grad", action="store_true", help="enable gradient normalization in reduced inversion checks")
    parser.add_argument("--case-inversion-forw-illumination", action="store_true", help="enable forward illumination preconditioning in reduced inversion checks")
    parser.add_argument("--case-inversion-auto-update-rho", action="store_true", help="enable rho auto update in reduced inversion checks")
    parser.add_argument("--case-inversion-waveform-normalize", action="store_true", help="enable waveform normalization in reduced inversion checks")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.example_gradient_processors is None:
        args.example_gradient_processors = [args.example_gradient_processor]

    reports = []
    for suite in args.suites:
        if suite == "compare-mini":
            reports.append(run_compare_mini(args))
        elif suite == "examples":
            reports.append(run_examples(args))
        elif suite == "case-checks":
            reports.append(run_case_checks(args))
        elif suite == "case-inversion":
            reports.append(run_case_inversions(args))
        else:
            reports.append(run_single_device_suite(suite, args))

    failed = [report for report in reports if report["status"] == "failed"]
    report = {
        "status": "failed" if failed else "ok",
        "suites": args.suites,
        "devices": args.devices,
        "prefer": args.prefer,
        "dtype": args.dtype,
        "summary": summarize_reports(reports),
        "reports": reports,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
