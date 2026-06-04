#!/usr/bin/env python
"""Run acoustic FWI baseline matrix for Deepwave-inspired optimization.

The matrix records production ADFWI iteration time and peak memory before any
new custom-operator implementation is introduced. It intentionally calls the
existing `acoustic_fwi_iteration_profile.py` benchmark so the timing breakdown
stays consistent with previous bv1.2 performance work.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "baselines"
)
VALIDATION_OUTPUT_BASES = {
    "reduced": REPO_ROOT
    / "examples"
    / "validation"
    / "marmousi2_acoustic_reduced"
    / "outputs"
    / "deepwave_baseline",
    "full_record": REPO_ROOT
    / "examples"
    / "validation"
    / "marmousi2_acoustic_full_record"
    / "outputs"
    / "deepwave_baseline",
}


def parse_int_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shots", default="1,3,40", help="Comma-separated shot counts.")
    parser.add_argument("--iterations", default="3,10", help="Comma-separated iteration counts.")
    parser.add_argument("--validation-case", default="full_record", choices=("reduced", "full_record"))
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--checkpoint-segments", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--case-output-root",
        type=Path,
        default=None,
        help="Directory containing observed data for the selected validation case.",
    )
    parser.add_argument("--dry-run", action="store_true")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def summarize_profile(report: Dict[str, Any]) -> Dict[str, Any]:
    iterations = report["iterations"]
    totals = [item["timing_total"] for item in iterations]
    forward = [item["timings"]["forward"] for item in iterations]
    backward = [item["timings"]["backward"] for item in iterations]
    loss_eval = [item["timings"]["loss_evaluation"] for item in iterations]
    grad_proc = [item["timings"]["gradient_processing"] for item in iterations]
    optimizer = [item["timings"]["optimizer_step"] for item in iterations]
    steady_totals = totals[1:] if len(totals) > 1 else totals
    return {
        "shots": report["shape"]["shots"],
        "iterations": len(iterations),
        "checkpoint_segments": report["shape"]["checkpoint_segments"],
        "batch_size": report["shape"]["batch_size"],
        "receivers": report["shape"]["receivers"],
        "nt": report["shape"]["nt"],
        "nx": report["shape"]["nx"],
        "nz": report["shape"]["nz"],
        "mean_iteration_seconds": mean(totals),
        "steady_mean_iteration_seconds": mean(steady_totals),
        "mean_forward_seconds": mean(forward),
        "mean_backward_seconds": mean(backward),
        "mean_loss_evaluation_seconds": mean(loss_eval),
        "mean_gradient_processing_seconds": mean(grad_proc),
        "mean_optimizer_seconds": mean(optimizer),
        "total_seconds": sum(totals),
        "peak_allocated_mib": report.get("memory", {}).get("peak_allocated_mib"),
        "losses": [item["loss"] for item in iterations],
        "all_finite": all(
            item["loss_finite"]
            and item["raw_grad_finite"]
            and item["grad_finite_after_processing"]
            and item["vp_finite_after_optimizer"]
            for item in iterations
        ),
        "vp_update_norm": report["vp_update_norm"],
    }


def profile_command(args: argparse.Namespace, *, shots: int, iterations: int, output_path: Path) -> List[str]:
    batch_size = args.batch_size if args.batch_size is not None else shots
    case_output_root = (
        args.case_output_root
        if args.case_output_root is not None
        else VALIDATION_OUTPUT_BASES[args.validation_case] / f"shot{shots}"
    )
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "benchmark" / "acoustic_fwi_iteration_profile.py"),
        "--validation-case",
        args.validation_case,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--shots",
        str(shots),
        "--iterations",
        str(iterations),
        "--checkpoint-segments",
        str(args.checkpoint_segments),
        "--batch-size",
        str(batch_size),
        "--output-root",
        str(case_output_root),
        "--result-json",
        str(output_path),
    ]
    if args.case_output_root is not None:
        command.append("--no-generate-observed")
    return command


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def markdown_table(rows: Iterable[Dict[str, Any]]) -> str:
    header = (
        "| Shots | Iterations | Checkpoints | Batch | Mean iter (s) | "
        "Steady mean (s) | Forward (s) | Backward (s) | Peak memory (MiB) | Finite |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    body = []
    for row in rows:
        peak = row["peak_allocated_mib"]
        body.append(
            "| {shots} | {iterations} | {checkpoint_segments} | {batch_size} | "
            "{mean_iteration_seconds:.4f} | {steady_mean_iteration_seconds:.4f} | "
            "{mean_forward_seconds:.4f} | {mean_backward_seconds:.4f} | "
            "{peak} | {finite} |".format(
                **row,
                peak="n/a" if peak is None else f"{peak:.2f}",
                finite="yes" if row["all_finite"] else "no",
            )
        )
    return header + "\n".join(body) + "\n"


def write_markdown(path: Path, *, payload: Dict[str, Any]) -> None:
    text = [
        "# Acoustic FWI Baseline Matrix",
        "",
        "This file records the production ADFWI baseline before Deepwave-inspired",
        "custom-operator work starts on this branch.",
        "",
        "## Configuration",
        "",
        f"- validation case: `{payload['config']['validation_case']}`",
        f"- device: `{payload['config']['device']}`",
        f"- dtype: `{payload['config']['dtype']}`",
        f"- checkpoint_segments: `{payload['config']['checkpoint_segments']}`",
        f"- generated at: `{payload['generated_at']}`",
        "",
        "## Results",
        "",
        markdown_table(payload["summary"]),
        "",
        "## Notes",
        "",
        "- `Mean iter` averages all iterations in the run.",
        "- `Steady mean` excludes the first iteration when more than one iteration is available.",
        "- Peak memory uses the active PyTorch backend memory API when available.",
        "- These runs are production baselines only; no Deepwave-inspired operator is used.",
        "",
    ]
    path.write_text("\n".join(text))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args(argv)

    shots_list = parse_int_list(args.shots)
    iterations_list = parse_int_list(args.iterations)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    runs = []
    for shots in shots_list:
        for iterations in iterations_list:
            output_path = raw_dir / f"production_shot{shots}_iter{iterations}_ckpt{args.checkpoint_segments}.json"
            command = profile_command(args, shots=shots, iterations=iterations, output_path=output_path)
            runs.append({"shots": shots, "iterations": iterations, "command": command, "output": str(output_path)})
            if args.dry_run:
                continue
            run_command(command)
            report = json.loads(output_path.read_text())
            summaries.append(summarize_profile(report))

    payload = {
        "status": "dry_run" if args.dry_run else "ok",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": {
            "validation_case": args.validation_case,
            "device": args.device,
            "dtype": args.dtype,
            "checkpoint_segments": args.checkpoint_segments,
            "shots": shots_list,
            "iterations": iterations_list,
            "batch_size": args.batch_size,
            "case_output_root": (
                str(args.case_output_root)
                if args.case_output_root is not None
                else str(VALIDATION_OUTPUT_BASES[args.validation_case] / "shot{shots}")
            ),
        },
        "runs": runs,
        "summary": summaries,
    }
    matrix_json = args.output_dir / "baseline-matrix-results.json"
    matrix_md = args.output_dir / "baseline-matrix-results.md"
    matrix_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(matrix_md, payload=payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
