#!/usr/bin/env python
"""Probe Python/NPU runtime visibility for the Ascend pressure-update custom op.

The compile gate proves the generated package can build. This runtime gate
installs that package into a temporary custom OPP path and checks whether it is
directly visible through PyTorch's NPU dispatch namespace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

from ascend_fused_pressure_update_prototype import DEFAULT_OUTPUT as DEFAULT_COMPILE_OUTPUT
from ascend_fused_pressure_update_prototype import run_probe as run_compile_probe
from ascend_fused_pressure_update_prototype import build_parser as build_compile_parser
from ascend_custom_op_scaffold_probe import REPO_ROOT


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "ascend_custom_op_runtime_probe_20260601.json"
)


def run_command(command, *, env=None, timeout=120) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": [str(item) for item in command],
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def compile_pressure_update_package(workspace: Path, timeout: int) -> Dict[str, Any]:
    parser = build_compile_parser()
    args = parser.parse_args(
        [
            "--workspace",
            str(workspace / "compile"),
            "--output",
            str(DEFAULT_COMPILE_OUTPUT),
            "--compile",
            "--compile-timeout",
            str(timeout),
        ]
    )
    report = run_compile_probe(args)
    package = workspace / "compile" / "out" / "build_out" / "custom_opp_ubuntu_aarch64.run"
    return {
        "report": report,
        "package": str(package),
        "package_exists": package.exists(),
    }


def install_package(package: Path, install_root: Path, timeout: int) -> Dict[str, Any]:
    install_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = str(install_root)
    return run_command(
        [
            str(package),
            "--quiet",
            f"--install-path={install_root}",
        ],
        env=env,
        timeout=timeout,
    )


def probe_torch_visibility(install_root: Path, timeout: int) -> Dict[str, Any]:
    vendor_root = install_root / "vendors" / "customize"
    env = os.environ.copy()
    env["ASCEND_CUSTOM_OPP_PATH"] = f"{vendor_root}:{env.get('ASCEND_CUSTOM_OPP_PATH', '')}"
    env["LD_LIBRARY_PATH"] = f"{vendor_root / 'op_api' / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    code = r'''
import json
import os
import torch
import torch_npu

target_names = [
    "FusedPressureUpdateForward",
    "fused_pressure_update_forward",
    "aclnn_fused_pressure_update_forward",
]
all_ops = torch._C._dispatch_get_all_op_names()
result = {
    "ASCEND_CUSTOM_OPP_PATH": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
    "LD_LIBRARY_PATH_contains_custom": "customize/op_api/lib" in os.environ.get("LD_LIBRARY_PATH", ""),
    "dispatch_matches": [name for name in all_ops if "FusedPressure" in name or "fused_pressure" in name],
    "torch_ops_npu_matches": {},
}
for name in target_names:
    value = getattr(torch.ops.npu, name, None)
    result["torch_ops_npu_matches"][name] = None if value is None else str(value)
print("ADFWI_RUNTIME_PROBE_JSON=" + json.dumps(result, sort_keys=True))
'''
    return run_command([sys.executable, "-c", code], env=env, timeout=timeout)


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="adfwi_custom_op_runtime_"))
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)
    compile_report = compile_pressure_update_package(workspace, args.compile_timeout)
    package = Path(compile_report["package"])

    install_report = None
    visibility_report = None
    status = "ok"
    if not compile_report["package_exists"] or compile_report["report"]["status"] != "ok":
        status = "compile_failed"
    else:
        install_root = workspace / "install"
        install_report = install_package(package, install_root, args.install_timeout)
        if install_report["returncode"] != 0:
            status = "install_failed"
        else:
            visibility_report = probe_torch_visibility(install_root, args.runtime_timeout)
            if visibility_report["returncode"] != 0:
                status = "runtime_probe_failed"
            else:
                visibility = {}
                for line in visibility_report["output"].splitlines():
                    if line.startswith("ADFWI_RUNTIME_PROBE_JSON="):
                        visibility = json.loads(line.split("=", 1)[1])
                        break
                if not visibility.get("dispatch_matches") and not any(
                    visibility.get("torch_ops_npu_matches", {}).values()
                ):
                    status = "not_exposed_to_torch_ops"

    return {
        "status": status,
        "purpose": "runtime visibility gate for generated Ascend pressure-update custom op",
        "workspace": str(workspace),
        "compile": compile_report,
        "install": install_report,
        "torch_visibility": visibility_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--install-timeout", type=int, default=120)
    parser.add_argument("--runtime-timeout", type=int, default=120)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "workspace": report["workspace"],
                "compile_status": report["compile"]["report"]["status"],
                "package_exists": report["compile"]["package_exists"],
                "install_returncode": None if report["install"] is None else report["install"]["returncode"],
                "runtime_returncode": None
                if report["torch_visibility"] is None
                else report["torch_visibility"]["returncode"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] in {"ok", "not_exposed_to_torch_ops"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
