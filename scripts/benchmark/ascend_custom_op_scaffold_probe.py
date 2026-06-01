#!/usr/bin/env python
"""Probe Ascend custom-op scaffolding for fused acoustic stencil work.

The script is intentionally limited to project scaffolding and environment
checks. It does not implement or load a production operator.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "ascend_custom_op_scaffold_probe_20260601.json"
)
DEFAULT_MSOPGEN = Path("/usr/local/Ascend/ascend-toolkit/latest/bin/msopgen")
DEFAULT_CANN = Path("/usr/local/Ascend/ascend-toolkit/latest")


def module_available(module_name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except ModuleNotFoundError:
        return False


def run_command(command, *, cwd: Optional[Path] = None, timeout: int = 120) -> Dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd is not None else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    return {
        "command": [str(item) for item in command],
        "cwd": str(cwd) if cwd is not None else None,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def write_ir_json(path: Path) -> None:
    payload = [
        {
            "op": "FusedPressureUpdateForward",
            "input_desc": [
                {"name": "p", "param_type": "required", "format": ["ND"], "type": ["fp32"]},
                {"name": "u", "param_type": "required", "format": ["ND"], "type": ["fp32"]},
                {"name": "w", "param_type": "required", "format": ["ND"], "type": ["fp32"]},
                {"name": "kappa1", "param_type": "required", "format": ["ND"], "type": ["fp32"]},
                {"name": "alpha1", "param_type": "required", "format": ["ND"], "type": ["fp32"]},
            ],
            "output_desc": [
                {"name": "p_next", "param_type": "required", "format": ["ND"], "type": ["fp32"]}
            ],
            "attr": [
                {"name": "free_surface_start", "param_type": "required", "type": "int"}
            ],
        }
    ]
    path.write_text(json.dumps(payload, indent=2) + "\n")
    path.chmod(0o600)


def patch_generated_python(project_dir: Path, python_executable: Path) -> Dict[str, Any]:
    changed = []
    replacements = {
        '"value": "python3"': f'"value": "{python_executable}"',
        "opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)": (
            f"opts=$({python_executable} $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)"
        ),
        "export HI_PYTHON=python3": f"export HI_PYTHON={python_executable}",
    }
    for path in [
        project_dir / "CMakePresets.json",
        project_dir / "build.sh",
        project_dir / "cmake" / "util" / "ascendc_compile_kernel.py",
    ]:
        if not path.exists():
            continue
        text = path.read_text()
        original = text
        for old, new in replacements.items():
            text = text.replace(old, new)
        if text != original:
            path.write_text(text)
            changed.append(str(path.relative_to(project_dir)))
    return {"changed_files": changed}


def inspect_generated_files(project_dir: Path) -> Dict[str, Any]:
    files = sorted(
        str(path.relative_to(project_dir))
        for path in project_dir.rglob("*")
        if path.is_file()
    )
    required = [
        "CMakeLists.txt",
        "CMakePresets.json",
        "build.sh",
        "op_host/fused_pressure_update_forward.cpp",
        "op_host/fused_pressure_update_forward_tiling.h",
        "op_kernel/fused_pressure_update_forward.cpp",
    ]
    return {
        "file_count": len(files),
        "first_files": files[:80],
        "required_files_present": {item: item in files for item in required},
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    python_executable = Path(args.python_executable).resolve()
    msopgen = Path(args.msopgen).resolve()
    cann_path = Path(args.cann_path).resolve()
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="adfwi_msopgen_probe_"))
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)
    ir_path = workspace / "fused_pressure_update_forward.json"
    project_dir = workspace / "out"
    write_ir_json(ir_path)

    dependency_checks = {
        "google": module_available("google"),
        "google.protobuf": module_available("google.protobuf"),
        "numpy": module_available("numpy"),
    }
    environment = {
        "python_executable": str(python_executable),
        "python_version": sys.version.split()[0],
        "msopgen": str(msopgen),
        "msopgen_exists": msopgen.exists(),
        "cann_path": str(cann_path),
        "cann_path_exists": cann_path.exists(),
        "npu_smi": shutil.which("npu-smi"),
    }

    gen_command = [
        str(msopgen),
        "gen",
        "-i",
        str(ir_path),
        "-f",
        "pytorch",
        "-c",
        args.compute_unit,
        "-out",
        str(project_dir),
        "-op",
        "FusedPressureUpdateForward",
        "-lan",
        "cpp",
    ]
    generation = run_command(gen_command, timeout=args.timeout)
    generated = project_dir.exists()
    patch_report = patch_generated_python(project_dir, python_executable) if generated else {"changed_files": []}
    generated_files = inspect_generated_files(project_dir) if generated else None

    compile_report = None
    compile_skipped_reason = None
    if args.compile:
        if not dependency_checks["google.protobuf"]:
            compile_skipped_reason = "missing google.protobuf in the active Python environment"
        elif generation["returncode"] != 0:
            compile_skipped_reason = "msopgen generation failed"
        else:
            command = ["bash", "build.sh"]
            env = os.environ.copy()
            env["ASCEND_HOME_PATH"] = str(cann_path)
            completed = subprocess.run(
                command,
                cwd=str(project_dir),
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.compile_timeout,
                check=False,
            )
            compile_report = {
                "command": command,
                "cwd": str(project_dir),
                "returncode": completed.returncode,
                "output": completed.stdout,
            }
    else:
        compile_skipped_reason = "compile flag not set"

    status = "ok"
    if generation["returncode"] != 0:
        status = "generation_failed"
    elif args.compile and compile_report is not None and compile_report["returncode"] != 0:
        status = "compile_failed"
    elif args.compile and compile_skipped_reason:
        status = "compile_skipped"

    return {
        "status": status,
        "purpose": "Ascend custom-op scaffold probe for fused acoustic pressure update",
        "workspace": str(workspace),
        "environment": environment,
        "dependency_checks": dependency_checks,
        "ir_json": str(ir_path),
        "generation": generation,
        "patch_generated_python": patch_report,
        "generated_files": generated_files,
        "compile_requested": args.compile,
        "compile_skipped_reason": compile_skipped_reason,
        "compile": compile_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--msopgen", type=Path, default=DEFAULT_MSOPGEN)
    parser.add_argument("--cann-path", type=Path, default=DEFAULT_CANN)
    parser.add_argument("--compute-unit", default="ai_core-ascend910b")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--compile-timeout", type=int, default=240)
    parser.add_argument("--compile", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    report = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"],
        "workspace": report["workspace"],
        "dependency_checks": report["dependency_checks"],
        "generated": report["generated_files"] is not None,
        "compile_requested": report["compile_requested"],
        "compile_skipped_reason": report["compile_skipped_reason"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0 if report["status"] in {"ok", "compile_skipped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
