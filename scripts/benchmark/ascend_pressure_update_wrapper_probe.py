#!/usr/bin/env python
"""Probe whether the pressure-update custom op is callable from PyTorch.

The previous gates proved that the AscendC kernel package can be generated,
compiled, installed, and that its ACLNN symbols exist in ``libcust_opapi.so``.
This gate adds the missing layer: a minimal PyTorch NPU C++ extension that
accepts ``torch.Tensor`` inputs and calls the generated ACLNN API through the
torch_npu op-plugin helper macros.

This script is intentionally standalone. It does not modify the production
propagator path.
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

from ascend_custom_op_scaffold_probe import REPO_ROOT
from ascend_fused_pressure_update_prototype import build_parser as build_compile_parser
from ascend_fused_pressure_update_prototype import run_probe as run_compile_probe
from ascend_custom_op_runtime_probe import install_package


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "ascend_pressure_update_wrapper_probe_20260601.json"
)


EXTENSION_CPP = r'''
#include <torch/extension.h>

#include "op_plugin/utils/op_api_common.h"
#include "torch_npu/csrc/framework/utils/OpPreparation.h"

at::Tensor fused_pressure_update_forward_wrapper(
    const at::Tensor& p,
    const at::Tensor& u,
    const at::Tensor& w,
    const at::Tensor& kappa1,
    const at::Tensor& alpha1,
    int64_t free_surface_start) {
  TORCH_CHECK(p.dim() == 3, "p must have shape [shot, nz_pml, nx_pml]");
  TORCH_CHECK(u.sizes() == p.sizes(), "u must match p shape");
  TORCH_CHECK(w.sizes() == p.sizes(), "w must match p shape");
  TORCH_CHECK(kappa1.dim() == 2, "kappa1 must have shape [nz_pml, nx_pml]");
  TORCH_CHECK(alpha1.sizes() == kappa1.sizes(), "alpha1 must match kappa1 shape");
  TORCH_CHECK(kappa1.size(0) == p.size(1) && kappa1.size(1) == p.size(2),
              "model coefficient shape must match p spatial shape");
  TORCH_CHECK(p.scalar_type() == at::kFloat, "prototype supports float32 only");

  auto out = at_npu::native::OpPreparation::apply_tensor_without_format(p);
  EXEC_NPU_CMD_EXT(aclnnFusedPressureUpdateForward, p, u, w, kappa1, alpha1, free_surface_start, out);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_pressure_update_forward", &fused_pressure_update_forward_wrapper,
        "Fused pressure update forward wrapper probe");
}
'''


SETUP_PY = r'''
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension
from torch_npu.utils.cpp_extension import NpuExtension
import os
import torch_npu

torch_npu_root = os.path.dirname(os.path.realpath(torch_npu.__file__))

setup(
    name="adfwi_pressure_update_wrapper_probe",
    ext_modules=[
        NpuExtension(
            name="adfwi_pressure_update_wrapper_probe",
            sources=["pressure_update_wrapper.cpp"],
            include_dirs=[
                os.path.join(torch_npu_root, "include", "third_party", "op-plugin"),
            ],
            extra_compile_args=["-O0", "-std=c++17"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
'''


RUNTIME_PY = r'''
import json
import os
import sys
from pathlib import Path

print("ADFWI_WRAPPER_STAGE=start", flush=True)
import torch
import torch_npu
print("ADFWI_WRAPPER_STAGE=imported_torch", flush=True)

sys.path.insert(0, str(Path(os.environ["ADFWI_WRAPPER_BUILD_DIR"])))
import adfwi_pressure_update_wrapper_probe as wrapper
print("ADFWI_WRAPPER_STAGE=imported_wrapper", flush=True)


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


torch.manual_seed(20260601)
device = "npu:0"
shape = tuple(int(item) for item in os.environ.get("ADFWI_RUNTIME_SHAPE", "2,9,10").split(","))
free_surface_start = 2
p_cpu = torch.randn(shape, dtype=torch.float32)
u_cpu = torch.randn(shape, dtype=torch.float32)
w_cpu = torch.randn(shape, dtype=torch.float32)
kappa_cpu = torch.rand(shape[1:], dtype=torch.float32) * 0.1
alpha_cpu = torch.rand(shape[1:], dtype=torch.float32) * 0.2

kernel_mode = os.environ.get("ADFWI_KERNEL_MODE", "pressure")
if kernel_mode.startswith("copy"):
    expected = p_cpu.clone()
else:
    expected = reference(p_cpu, u_cpu, w_cpu, kappa_cpu, alpha_cpu, free_surface_start)
print("ADFWI_WRAPPER_STAGE=prepared_cpu_reference", flush=True)

p = p_cpu.to(device)
u = u_cpu.to(device)
w = w_cpu.to(device)
kappa = kappa_cpu.to(device)
alpha = alpha_cpu.to(device)
torch.npu.synchronize()
print("ADFWI_WRAPPER_STAGE=moved_to_npu", flush=True)

actual = wrapper.fused_pressure_update_forward(p, u, w, kappa, alpha, free_surface_start)
torch.npu.synchronize()
print("ADFWI_WRAPPER_STAGE=custom_op_completed", flush=True)
actual_cpu = actual.cpu()
diff = (actual_cpu - expected).abs()

result = {
    "device": device,
    "shape": list(shape),
    "kernel_mode": kernel_mode,
    "free_surface_start": free_surface_start,
    "output_shape": list(actual_cpu.shape),
    "max_abs_diff": float(diff.max().item()),
    "allclose_atol0_rtol0": bool(torch.allclose(actual_cpu, expected, atol=0.0, rtol=0.0)),
    "allclose_atol1e_6_rtol1e_6": bool(torch.allclose(actual_cpu, expected, atol=1e-6, rtol=1e-6)),
}
print("ADFWI_WRAPPER_PROBE_JSON=" + json.dumps(result, sort_keys=True))
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
            "output_tail": "\n".join(completed.stdout.splitlines()[-120:]),
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
            "output_tail": "\n".join(output.splitlines()[-120:]),
        }


def build_wrapper(wrapper_dir: Path, env: Dict[str, str], timeout: int) -> Dict[str, Any]:
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    (wrapper_dir / "pressure_update_wrapper.cpp").write_text(EXTENSION_CPP)
    (wrapper_dir / "setup.py").write_text(SETUP_PY)
    return run_command(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=wrapper_dir,
        env=env,
        timeout=timeout,
    )


def run_wrapper(wrapper_dir: Path, env: Dict[str, str], timeout: int) -> Dict[str, Any]:
    runtime_script = wrapper_dir / "run_wrapper_probe.py"
    runtime_script.write_text(RUNTIME_PY)
    runtime_env = env.copy()
    runtime_env["ADFWI_WRAPPER_BUILD_DIR"] = str(wrapper_dir)
    report = run_command([sys.executable, str(runtime_script)], env=runtime_env, timeout=timeout)
    parsed = None
    for line in report["output_tail"].splitlines():
        if line.startswith("ADFWI_WRAPPER_PROBE_JSON="):
            parsed = json.loads(line.split("=", 1)[1])
            break
    report["parsed_result"] = parsed
    return report


def compile_pressure_update_package(workspace: Path, timeout: int, *, kernel_mode: str, block_dim: int) -> Dict[str, Any]:
    parser = build_compile_parser()
    args = parser.parse_args(
        [
            "--workspace",
            str(workspace / "compile"),
            "--compile",
            "--compile-timeout",
            str(timeout),
            "--kernel-mode",
            kernel_mode,
            "--block-dim",
            str(block_dim),
        ]
    )
    report = run_compile_probe(args)
    package = workspace / "compile" / "out" / "build_out" / "custom_opp_ubuntu_aarch64.run"
    return {
        "report": report,
        "package": str(package),
        "package_exists": package.exists(),
    }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="adfwi_pressure_wrapper_"))
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)

    compile_report = compile_pressure_update_package(
        workspace,
        args.compile_timeout,
        kernel_mode=args.kernel_mode,
        block_dim=args.block_dim,
    )
    status = "ok"
    install_report = None
    wrapper_build = None
    wrapper_runtime = None
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
            env["ADFWI_KERNEL_MODE"] = args.kernel_mode
            env["ADFWI_RUNTIME_SHAPE"] = ",".join(str(item) for item in args.runtime_shape)
            wrapper_dir = workspace / "wrapper"
            wrapper_build = build_wrapper(wrapper_dir, env, args.wrapper_build_timeout)
            if wrapper_build.get("timed_out"):
                status = "wrapper_build_timed_out"
            elif wrapper_build["returncode"] != 0:
                status = "wrapper_build_failed"
            else:
                wrapper_runtime = run_wrapper(wrapper_dir, env, args.runtime_timeout)
                if wrapper_runtime["returncode"] != 0:
                    status = "wrapper_runtime_failed"
                elif not wrapper_runtime.get("parsed_result", {}).get("allclose_atol1e_6_rtol1e_6", False):
                    status = "wrapper_numerical_mismatch"

    return {
        "status": status,
        "purpose": "minimal PyTorch NPU wrapper feasibility gate for Ascend pressure-update custom op",
        "kernel_mode": args.kernel_mode,
        "block_dim": args.block_dim,
        "runtime_shape": list(args.runtime_shape),
        "workspace": str(workspace),
        "compile": compile_report,
        "install": install_report,
        "wrapper_build": wrapper_build,
        "wrapper_runtime": wrapper_runtime,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--install-timeout", type=int, default=120)
    parser.add_argument("--wrapper-build-timeout", type=int, default=300)
    parser.add_argument("--runtime-timeout", type=int, default=120)
    parser.add_argument("--kernel-mode", choices=("pressure", "copy", "copy_vector"), default="pressure")
    parser.add_argument("--block-dim", type=int, default=8)
    parser.add_argument("--runtime-shape", type=int, nargs=3, default=(2, 9, 10))
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
                "kernel_mode": report["kernel_mode"],
                "block_dim": report["block_dim"],
                "runtime_shape": report["runtime_shape"],
                "custom_op_compile_status": report["compile"]["report"]["status"],
                "install_returncode": None if report["install"] is None else report["install"]["returncode"],
                "wrapper_build_returncode": None
                if report["wrapper_build"] is None
                else report["wrapper_build"]["returncode"],
                "wrapper_runtime_returncode": None
                if report["wrapper_runtime"] is None
                else report["wrapper_runtime"]["returncode"],
                "wrapper_result": None
                if report["wrapper_runtime"] is None
                else report["wrapper_runtime"].get("parsed_result"),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
