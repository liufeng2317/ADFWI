#!/usr/bin/env python
"""Run acoustic backend smoke tests across one or more devices.

This wrapper invokes ``acoustic_backend_smoke.py`` once per requested device and
prints a single JSON report. It writes no notebook, figure, or waveform output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

SCRIPT = Path(__file__).resolve().with_name("acoustic_backend_smoke.py")


def parse_devices(value: str) -> List[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices:
        raise argparse.ArgumentTypeError("at least one device is required")
    return devices


def extract_json(stdout: str) -> Dict[str, Any]:
    start = stdout.find("{")
    if start < 0:
        raise ValueError("child process did not print a JSON object")
    return json.loads(stdout[start:])


def run_one(device: str, args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(SCRIPT),
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
        "--nx",
        str(args.nx),
        "--nz",
        str(args.nz),
        "--nabc",
        str(args.nabc),
        "--nt",
        str(args.nt),
        "--dt",
        str(args.dt),
        "--f0",
        str(args.f0),
        "--dx",
        str(args.dx),
        "--dz",
        str(args.dz),
    ]
    if args.fallback_cpu:
        cmd.append("--fallback-cpu")
    if args.skip_backward:
        cmd.append("--skip-backward")

    proc = subprocess.run(cmd, cwd=str(args.repo_root), text=True, capture_output=True)
    result: Dict[str, Any] = {
        "device_request": device,
        "returncode": proc.returncode,
    }
    if proc.stdout.strip():
        try:
            result["result"] = extract_json(proc.stdout)
        except Exception as exc:
            result["stdout"] = proc.stdout
            result["parse_error"] = repr(exc)
    if proc.stderr.strip() and (args.include_stderr or proc.returncode != 0):
        result["stderr"] = proc.stderr

    if proc.returncode != 0 and not (args.skip_unavailable and proc.returncode == 2):
        result["status"] = "failed"
    elif proc.returncode == 2:
        result["status"] = "unavailable"
    else:
        result["status"] = "ok"
    return result


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run acoustic backend smoke tests for multiple devices.")
    parser.add_argument("--devices", type=parse_devices, default=parse_devices("cpu,npu:0"), help="comma-separated devices, e.g. cpu,npu:0 or cpu,auto")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--skip-unavailable", action="store_true", default=True, help="treat unavailable devices as skipped")
    parser.add_argument("--strict-unavailable", action="store_false", dest="skip_unavailable", help="fail if a requested device is unavailable")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    parser.add_argument("--skip-backward", action="store_true")
    parser.add_argument("--include-stderr", action="store_true", help="include child stderr in the JSON report even for successful runs")
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--nx", type=int, default=24)
    parser.add_argument("--nz", type=int, default=20)
    parser.add_argument("--nabc", type=int, default=4)
    parser.add_argument("--nt", type=int, default=30)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--f0", type=float, default=15.0)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.set_defaults(repo_root=repo_root)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    runs = [run_one(device, args) for device in args.devices]
    failed = [run for run in runs if run["status"] == "failed"]
    report = {
        "status": "failed" if failed else "ok",
        "script": str(SCRIPT.relative_to(args.repo_root)),
        "devices": args.devices,
        "runs": runs,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
