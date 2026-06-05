#!/usr/bin/env python
"""Benchmark the aligned Ascend pressure-update forward prototype.

This benchmark compares a single pressure update implemented in PyTorch tensor
operations with the experimental AscendC ``pressure_aligned_chunks`` custom op.
It is a forward-only feasibility benchmark and does not modify production
ADFWI propagator code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ascend_custom_op_scaffold_probe import REPO_ROOT
from ascend_custom_op_runtime_probe import install_package
from ascend_pressure_update_wrapper_probe import (
    build_wrapper,
    compile_pressure_update_package,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "ascend_pressure_update_forward_benchmark_20260605.json"
)


RUNTIME_BENCHMARK_PY = r'''
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(os.environ["ADFWI_WRAPPER_BUILD_DIR"])))
import adfwi_pressure_update_wrapper_probe as wrapper


def reference(p, u, w, kappa1, alpha1, free_surface_start):
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    nz_pml = p.shape[1]
    nx_pml = p.shape[2]
    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    div_p = (
        c1
        * (
            u[:, zp, 2 : nx_pml - 2]
            - u[:, zp, 1 : nx_pml - 3]
            + w[:, zp, 2 : nx_pml - 2]
            - w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2]
        )
        + c2
        * (
            u[:, zp, 3 : nx_pml - 1]
            - u[:, zp, 0 : nx_pml - 4]
            + w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2]
            - w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2]
        )
    )
    out = p.clone()
    out[:, zp, xp] = (1.0 - kappa1[zp, xp]) * p[:, zp, xp] - alpha1[zp, xp] * div_p
    return out


def synchronize():
    torch.npu.synchronize()


def time_call(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        out = fn()
        synchronize()
        elapsed.append(time.perf_counter() - start)
    return out, elapsed


def summarize(seconds):
    return {
        "mean_seconds": float(statistics.mean(seconds)),
        "median_seconds": float(statistics.median(seconds)),
        "min_seconds": float(min(seconds)),
        "max_seconds": float(max(seconds)),
        "repeats": len(seconds),
    }


def run_case(shape, warmup, repeats):
    torch.manual_seed(20260605)
    device = "npu:0"
    free_surface_start = 2
    p = torch.randn(shape, dtype=torch.float32, device=device)
    u = torch.randn(shape, dtype=torch.float32, device=device)
    w = torch.randn(shape, dtype=torch.float32, device=device)
    kappa1 = torch.rand(shape[1:], dtype=torch.float32, device=device) * 0.1
    alpha1 = torch.rand(shape[1:], dtype=torch.float32, device=device) * 0.2
    synchronize()

    custom_fn = lambda: wrapper.fused_pressure_update_forward(p, u, w, kappa1, alpha1, free_surface_start)
    torch_fn = lambda: reference(p, u, w, kappa1, alpha1, free_surface_start)

    custom_out, custom_seconds = time_call(custom_fn, warmup, repeats)
    torch_out, torch_seconds = time_call(torch_fn, warmup, repeats)
    diff = (custom_out - torch_out).abs()
    max_abs = float(diff.max().item())
    ref_max = float(torch_out.abs().max().item())

    custom_summary = summarize(custom_seconds)
    torch_summary = summarize(torch_seconds)
    return {
        "shape": list(shape),
        "device": device,
        "free_surface_start": free_surface_start,
        "warmup": warmup,
        "repeats": repeats,
        "custom": custom_summary,
        "torch_reference": torch_summary,
        "speedup_vs_torch_reference_median": float(
            torch_summary["median_seconds"] / custom_summary["median_seconds"]
        ),
        "max_abs_diff": max_abs,
        "relative_max_abs_diff": float(max_abs / ref_max) if ref_max != 0.0 else 0.0,
        "allclose_atol1e_6_rtol1e_6": bool(torch.allclose(custom_out, torch_out, atol=1e-6, rtol=1e-6)),
    }


shapes = [tuple(int(part) for part in item.split(",")) for item in os.environ["ADFWI_BENCHMARK_SHAPES"].split(";")]
warmup = int(os.environ["ADFWI_BENCHMARK_WARMUP"])
repeats = int(os.environ["ADFWI_BENCHMARK_REPEATS"])
results = [run_case(shape, warmup, repeats) for shape in shapes]
print("ADFWI_PRESSURE_FORWARD_BENCHMARK_JSON=" + json.dumps({"cases": results}, sort_keys=True))
'''


def run_command(command, *, cwd: Path | None = None, env: Dict[str, str] | None = None, timeout: int = 120) -> Dict[str, Any]:
    command = [str(item) for item in command]
    try:
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": command,
            "cwd": None if cwd is None else str(cwd),
            "returncode": completed.returncode,
            "timed_out": False,
            "timeout_seconds": timeout,
            "output_tail": "\n".join(completed.stdout.splitlines()[-160:]),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output = output or ""
        return {
            "command": command,
            "cwd": None if cwd is None else str(cwd),
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout,
            "output_tail": "\n".join(output.splitlines()[-160:]),
        }


def parse_shape(value: str) -> Tuple[int, int, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be formatted as shot,nz,nx")
    return tuple(int(part) for part in parts)  # type: ignore[return-value]


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="adfwi_pressure_forward_bench_"))
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)

    compile_report = compile_pressure_update_package(
        workspace,
        args.compile_timeout,
        kernel_mode="pressure_aligned_chunks",
        block_dim=args.block_dim,
    )
    status = "ok"
    install_report = None
    wrapper_build = None
    benchmark_runtime = None
    parsed = None
    if not compile_report["package_exists"] or compile_report["report"]["status"] != "ok":
        status = "custom_op_compile_failed"
    else:
        install_root = workspace / "install"
        install_report = install_package(Path(compile_report["package"]), install_root, args.install_timeout)
        if install_report["returncode"] != 0:
            status = "custom_op_install_failed"
        else:
            vendor_root = install_root / "vendors" / "customize"
            env = os.environ.copy()
            env["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor_root}:{env.get('ASCEND_CUSTOM_OPP_PATH', '')}"
            env["LD_LIBRARY_PATH"] = f"{vendor_root / 'op_api' / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
            env["ADFWI_BENCHMARK_SHAPES"] = ";".join(",".join(str(v) for v in shape) for shape in args.shape)
            env["ADFWI_BENCHMARK_WARMUP"] = str(args.warmup)
            env["ADFWI_BENCHMARK_REPEATS"] = str(args.repeats)
            wrapper_dir = workspace / "wrapper"
            wrapper_build = build_wrapper(wrapper_dir, env, args.wrapper_build_timeout)
            if wrapper_build.get("timed_out"):
                status = "wrapper_build_timed_out"
            elif wrapper_build["returncode"] != 0:
                status = "wrapper_build_failed"
            else:
                runtime_script = wrapper_dir / "run_forward_benchmark.py"
                runtime_script.write_text(RUNTIME_BENCHMARK_PY)
                runtime_env = env.copy()
                runtime_env["ADFWI_WRAPPER_BUILD_DIR"] = str(wrapper_dir)
                benchmark_runtime = run_command([sys.executable, str(runtime_script)], env=runtime_env, timeout=args.runtime_timeout)
                if benchmark_runtime["returncode"] != 0:
                    status = "benchmark_runtime_failed"
                else:
                    for line in benchmark_runtime["output_tail"].splitlines():
                        if line.startswith("ADFWI_PRESSURE_FORWARD_BENCHMARK_JSON="):
                            parsed = json.loads(line.split("=", 1)[1])
                            break
                    if parsed is None:
                        status = "benchmark_parse_failed"
                    elif not all(case["allclose_atol1e_6_rtol1e_6"] for case in parsed["cases"]):
                        status = "benchmark_numerical_mismatch"

    return {
        "status": status,
        "purpose": "forward-only benchmark for aligned Ascend pressure-update custom op",
        "kernel_mode": "pressure_aligned_chunks",
        "block_dim": args.block_dim,
        "workspace": str(workspace),
        "shapes": [list(shape) for shape in args.shape],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "compile": compile_report,
        "install": install_report,
        "wrapper_build": wrapper_build,
        "benchmark_runtime": benchmark_runtime,
        "results": parsed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--shape", type=parse_shape, action="append", default=[(3, 64, 80), (40, 64, 80)])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--install-timeout", type=int, default=120)
    parser.add_argument("--wrapper-build-timeout", type=int, default=300)
    parser.add_argument("--runtime-timeout", type=int, default=240)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        "status": report["status"],
        "output": str(args.output),
        "kernel_mode": report["kernel_mode"],
        "block_dim": report["block_dim"],
        "results": report["results"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
