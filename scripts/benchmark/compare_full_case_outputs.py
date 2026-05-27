#!/usr/bin/env python
"""Compare saved full-case Marmousi2 output summaries.

The full-case tests write one `summary.json` per run. This helper turns those
files into a stable comparison report so benchmark notes do not depend on
manual JSON parsing.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def resolve_summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "summary.json"
    return path


def load_summary(path: Path) -> Dict[str, Any]:
    summary_path = resolve_summary_path(path)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary file does not exist: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if summary.get("status") != "ok":
        raise ValueError(f"summary status is not ok: {summary_path}")
    return summary


def finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in summary: {value}")
    return number


def is_monotonic_nonincreasing(values: Iterable[float]) -> bool:
    values = list(values)
    return all(next_value <= value for value, next_value in zip(values, values[1:]))


def max_abs_diff(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) != len(right):
        return None
    if not left:
        return 0.0
    return max(abs(a - b) for a, b in zip(left, right))


def max_rel_diff(left: List[float], right: List[float]) -> Optional[float]:
    if len(left) != len(right):
        return None
    if not left:
        return 0.0
    diffs = []
    for a, b in zip(left, right):
        scale = max(abs(a), abs(b), 1.0)
        diffs.append(abs(a - b) / scale)
    return max(diffs)


def summarize_run(label: str, input_path: Path, summary: Dict[str, Any]) -> Dict[str, Any]:
    inversion = summary["inversion"]
    backend = summary.get("backend", {})
    observed = summary.get("observed", {})
    subset = summary.get("subset", {})
    loss_history = [finite_float(value) for value in inversion["loss_history"]]
    iterations = int(inversion["iterations"])
    seconds = finite_float(inversion["seconds"])

    return {
        "label": label,
        "input": str(input_path),
        "summary": str(resolve_summary_path(input_path)),
        "backend": {
            "name": backend.get("name"),
            "device": backend.get("device"),
            "dtype": backend.get("dtype"),
            "memory_allocated": backend.get("memory_allocated"),
        },
        "case": summary.get("case"),
        "seed": summary.get("seed"),
        "shot_count": subset.get("shot_count"),
        "nt_samples": subset.get("nt_samples"),
        "iterations": iterations,
        "checkpoint_segments": subset.get("checkpoint_segments"),
        "observed_forward_seconds": finite_float(observed.get("synthetic_true_forward_seconds")),
        "inversion_seconds": seconds,
        "seconds_per_iteration": seconds / iterations,
        "initial_loss": finite_float(inversion.get("initial_loss")),
        "final_loss": finite_float(inversion.get("loss")),
        "loss_delta": finite_float(inversion.get("loss_delta")),
        "loss_relative_delta": finite_float(inversion.get("loss_relative_delta")),
        "loss_min": finite_float(inversion.get("loss_min")),
        "loss_max": finite_float(inversion.get("loss_max")),
        "loss_history": loss_history,
        "loss_monotonic_nonincreasing": is_monotonic_nonincreasing(loss_history),
        "gradient_processor": inversion.get("gradient_processor"),
        "vp_grad_norm": finite_float(inversion.get("vp_grad_norm")),
        "vp_update_norm": finite_float(inversion.get("vp_update_norm")),
    }


def numeric_diff(reference: Optional[float], value: Optional[float]) -> Dict[str, Optional[float]]:
    if reference is None or value is None:
        return {"reference": reference, "value": value, "abs_diff": None, "rel_diff": None}
    scale = max(abs(reference), abs(value), 1.0)
    return {
        "reference": reference,
        "value": value,
        "abs_diff": abs(value - reference),
        "rel_diff": abs(value - reference) / scale,
    }


def compare_runs(reference: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    left = reference["loss_history"]
    right = run["loss_history"]
    return {
        "reference_label": reference["label"],
        "label": run["label"],
        "seconds_per_iteration": numeric_diff(reference["seconds_per_iteration"], run["seconds_per_iteration"]),
        "final_loss": numeric_diff(reference["final_loss"], run["final_loss"]),
        "loss_delta": numeric_diff(reference["loss_delta"], run["loss_delta"]),
        "loss_relative_delta": numeric_diff(reference["loss_relative_delta"], run["loss_relative_delta"]),
        "vp_grad_norm": numeric_diff(reference["vp_grad_norm"], run["vp_grad_norm"]),
        "vp_update_norm": numeric_diff(reference["vp_update_norm"], run["vp_update_norm"]),
        "loss_history": {
            "length_match": len(left) == len(right),
            "reference_length": len(left),
            "length": len(right),
            "max_abs_diff": max_abs_diff(left, right),
            "max_rel_diff": max_rel_diff(left, right),
        },
    }


def build_report(paths: List[Path], labels: Optional[List[str]], reference_index: int) -> Dict[str, Any]:
    if labels is not None and len(labels) != len(paths):
        raise ValueError("--labels must have the same number of entries as paths")
    if reference_index < 0 or reference_index >= len(paths):
        raise ValueError("--reference-index is out of range")

    runs = []
    for index, path in enumerate(paths):
        label = labels[index] if labels is not None else path.parent.name if path.name == "summary.json" else path.name
        runs.append(summarize_run(label, path, load_summary(path)))

    reference = runs[reference_index]
    comparisons = [compare_runs(reference, run) for index, run in enumerate(runs) if index != reference_index]
    return {
        "status": "ok",
        "reference_label": reference["label"],
        "runs": runs,
        "comparisons": comparisons,
    }


def parse_labels(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    labels = [item.strip() for item in value.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("--labels must contain at least one label")
    return labels


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Run output directories or summary.json files")
    parser.add_argument("--labels", type=parse_labels, help="Comma-separated labels matching paths")
    parser.add_argument("--reference-index", type=int, default=0, help="Reference run index for comparisons")
    parser.add_argument("--fail-on-loss-drift", action="store_true", help="Exit nonzero when loss drift exceeds tolerances")
    parser.add_argument("--loss-abs-tol", type=float, default=1e-5)
    parser.add_argument("--loss-rel-tol", type=float, default=1e-8)
    return parser.parse_args(argv)


def loss_drift_failed(report: Dict[str, Any], abs_tol: float, rel_tol: float) -> bool:
    for comparison in report["comparisons"]:
        final_loss = comparison["final_loss"]
        history = comparison["loss_history"]
        if final_loss["abs_diff"] is not None and final_loss["abs_diff"] > max(abs_tol, rel_tol * max(abs(final_loss["reference"]), 1.0)):
            return True
        if history["max_abs_diff"] is not None and history["max_abs_diff"] > abs_tol:
            return True
        if history["max_rel_diff"] is not None and history["max_rel_diff"] > rel_tol:
            return True
    return False


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    report = build_report(args.paths, args.labels, args.reference_index)
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.fail_on_loss_drift and loss_drift_failed(report, args.loss_abs_tol, args.loss_rel_tol):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
