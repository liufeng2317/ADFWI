#!/usr/bin/env python
"""Compile a minimal AscendC pressure-update custom-op prototype.

This script is a feasibility gate for the fused acoustic stencil line. It
generates an Ascend custom-op scaffold, replaces the placeholder kernel with a
minimal pressure-update implementation, compiles the package, and records the
result. It does not install or connect the operator to ADFWI production code.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ascend_custom_op_scaffold_probe import (
    DEFAULT_CANN,
    DEFAULT_MSOPGEN,
    REPO_ROOT,
    inspect_generated_files,
    patch_generated_python,
    run_command,
    write_ir_json,
)


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "ascend_fused_pressure_update_compile_20260601.json"
)


TILING_HEADER = r'''
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedPressureUpdateForwardTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF(uint32_t, src_n);
  TILING_DATA_FIELD_DEF(uint32_t, nz_pml);
  TILING_DATA_FIELD_DEF(uint32_t, nx_pml);
  TILING_DATA_FIELD_DEF(uint32_t, free_surface_start);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedPressureUpdateForward, FusedPressureUpdateForwardTilingData)
}
'''


KERNEL_CPP = r'''
#include "kernel_operator.h"

using namespace AscendC;

class KernelFusedPressureUpdateForward {
public:
    __aicore__ inline KernelFusedPressureUpdateForward() {}

    __aicore__ inline void Init(GM_ADDR p, GM_ADDR u, GM_ADDR w, GM_ADDR kappa1, GM_ADDR alpha1,
                                GM_ADDR p_next, uint32_t size, uint32_t src_n, uint32_t nz_pml,
                                uint32_t nx_pml, uint32_t free_surface_start) {
        this->size = size;
        this->src_n = src_n;
        this->nz_pml = nz_pml;
        this->nx_pml = nx_pml;
        this->free_surface_start = free_surface_start;
        p_gm.SetGlobalBuffer((__gm__ float*)p, size);
        u_gm.SetGlobalBuffer((__gm__ float*)u, size);
        w_gm.SetGlobalBuffer((__gm__ float*)w, size);
        kappa1_gm.SetGlobalBuffer((__gm__ float*)kappa1, nz_pml * nx_pml);
        alpha1_gm.SetGlobalBuffer((__gm__ float*)alpha1, nz_pml * nx_pml);
        p_next_gm.SetGlobalBuffer((__gm__ float*)p_next, size);
    }

    __aicore__ inline void Process() {
        const uint32_t block_idx = GetBlockIdx();
        const uint32_t block_num = GetBlockNum();
        const uint32_t elems_per_block = (size + block_num - 1) / block_num;
        const uint32_t begin = block_idx * elems_per_block;
        uint32_t end = begin + elems_per_block;
        if (end > size) {
            end = size;
        }

        constexpr float c1 = 9.0f / 8.0f;
        constexpr float c2 = -1.0f / 24.0f;
        const uint32_t plane = nz_pml * nx_pml;

        for (uint32_t index = begin; index < end; ++index) {
            const uint32_t local = index % plane;
            const uint32_t z = local / nx_pml;
            const uint32_t x = local % nx_pml;
            float value = p_gm.GetValue(index);

            if (z >= free_surface_start + 1 && z < nz_pml - 2 && x >= 2 && x < nx_pml - 2) {
                const float div_p =
                    c1 * (u_gm.GetValue(index) - u_gm.GetValue(index - 1) +
                          w_gm.GetValue(index) - w_gm.GetValue(index - nx_pml)) +
                    c2 * (u_gm.GetValue(index + 1) - u_gm.GetValue(index - 2) +
                          w_gm.GetValue(index + nx_pml) - w_gm.GetValue(index - 2 * nx_pml));
                const uint32_t model_index = z * nx_pml + x;
                value = (1.0f - kappa1_gm.GetValue(model_index)) * value -
                        alpha1_gm.GetValue(model_index) * div_p;
            }
            p_next_gm.SetValue(index, value);
        }
    }

private:
    uint32_t size;
    uint32_t src_n;
    uint32_t nz_pml;
    uint32_t nx_pml;
    uint32_t free_surface_start;
    GlobalTensor<float> p_gm;
    GlobalTensor<float> u_gm;
    GlobalTensor<float> w_gm;
    GlobalTensor<float> kappa1_gm;
    GlobalTensor<float> alpha1_gm;
    GlobalTensor<float> p_next_gm;
};

extern "C" __global__ __aicore__ void fused_pressure_update_forward(
    GM_ADDR p, GM_ADDR u, GM_ADDR w, GM_ADDR kappa1, GM_ADDR alpha1,
    GM_ADDR p_next, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedPressureUpdateForward op;
    op.Init(p, u, w, kappa1, alpha1, p_next, tiling_data.size, tiling_data.src_n,
            tiling_data.nz_pml, tiling_data.nx_pml, tiling_data.free_surface_start);
    op.Process();
}
'''

COPY_KERNEL_CPP = r'''
#include "kernel_operator.h"

using namespace AscendC;

class KernelFusedPressureUpdateForwardCopy {
public:
    __aicore__ inline KernelFusedPressureUpdateForwardCopy() {}

    __aicore__ inline void Init(GM_ADDR p, GM_ADDR p_next, uint32_t size) {
        this->size = size;
        p_gm.SetGlobalBuffer((__gm__ float*)p, size);
        p_next_gm.SetGlobalBuffer((__gm__ float*)p_next, size);
    }

    __aicore__ inline void Process() {
        const uint32_t block_idx = GetBlockIdx();
        const uint32_t block_num = GetBlockNum();
        const uint32_t elems_per_block = (size + block_num - 1) / block_num;
        const uint32_t begin = block_idx * elems_per_block;
        uint32_t end = begin + elems_per_block;
        if (end > size) {
            end = size;
        }

        for (uint32_t index = begin; index < end; ++index) {
            p_next_gm.SetValue(index, p_gm.GetValue(index));
        }
    }

private:
    uint32_t size;
    GlobalTensor<float> p_gm;
    GlobalTensor<float> p_next_gm;
};

extern "C" __global__ __aicore__ void fused_pressure_update_forward(
    GM_ADDR p, GM_ADDR u, GM_ADDR w, GM_ADDR kappa1, GM_ADDR alpha1,
    GM_ADDR p_next, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelFusedPressureUpdateForwardCopy op;
    op.Init(p, p_next, tiling_data.size);
    op.Process();
}
'''


HOST_TILING_OLD = '''  FusedPressureUpdateForwardTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  context->SetBlockDim(8);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
'''


HOST_TILING_NEW_TEMPLATE = '''  FusedPressureUpdateForwardTilingData tiling;
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  const gert::Shape storage_shape = x1_shape->GetStorageShape();
  int32_t data_sz = 1;
  for (int i = 0; i < storage_shape.GetDimNum(); i++) {
    data_sz *= storage_shape.GetDim(i);
  }
  const int64_t* free_surface_start = context->GetAttrs()->GetAttrPointer<int64_t>(0);
  tiling.set_size(data_sz);
  tiling.set_src_n(static_cast<uint32_t>(storage_shape.GetDim(0)));
  tiling.set_nz_pml(static_cast<uint32_t>(storage_shape.GetDim(1)));
  tiling.set_nx_pml(static_cast<uint32_t>(storage_shape.GetDim(2)));
  tiling.set_free_surface_start(static_cast<uint32_t>(*free_surface_start));
  context->SetBlockDim(__BLOCK_DIM__);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
'''


def pressure_update_reference(
    p: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    kappa1: torch.Tensor,
    alpha1: torch.Tensor,
    *,
    free_surface_start: int,
) -> torch.Tensor:
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


def direct_scalar_reference(
    p: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    kappa1: torch.Tensor,
    alpha1: torch.Tensor,
    *,
    free_surface_start: int,
) -> torch.Tensor:
    c1 = 9.0 / 8.0
    c2 = -1.0 / 24.0
    src_n, nz_pml, nx_pml = p.shape
    out = p.clone()
    for ishot in range(src_n):
        for z in range(nz_pml):
            for x in range(nx_pml):
                if z >= free_surface_start + 1 and z < nz_pml - 2 and x >= 2 and x < nx_pml - 2:
                    div_p = (
                        c1 * (u[ishot, z, x] - u[ishot, z, x - 1] + w[ishot, z, x] - w[ishot, z - 1, x])
                        + c2
                        * (
                            u[ishot, z, x + 1]
                            - u[ishot, z, x - 2]
                            + w[ishot, z + 1, x]
                            - w[ishot, z - 2, x]
                        )
                    )
                    out[ishot, z, x] = (1.0 - kappa1[z, x]) * p[ishot, z, x] - alpha1[z, x] * div_p
    return out


def run_reference_contract() -> Dict[str, Any]:
    torch.manual_seed(20260601)
    shape = (2, 9, 10)
    model_shape = shape[1:]
    p = torch.randn(shape, dtype=torch.float32)
    u = torch.randn(shape, dtype=torch.float32)
    w = torch.randn(shape, dtype=torch.float32)
    kappa1 = torch.rand(model_shape, dtype=torch.float32) * 0.1
    alpha1 = torch.rand(model_shape, dtype=torch.float32) * 0.2
    free_surface_start = 2
    vector = pressure_update_reference(p, u, w, kappa1, alpha1, free_surface_start=free_surface_start)
    scalar = direct_scalar_reference(p, u, w, kappa1, alpha1, free_surface_start=free_surface_start)
    diff = (vector - scalar).abs()
    return {
        "shape": list(shape),
        "free_surface_start": free_surface_start,
        "max_abs_diff": float(diff.max().item()),
        "allclose": bool(torch.allclose(vector, scalar, atol=0.0, rtol=0.0)),
    }


def patch_pressure_update_project(project_dir: Path, *, kernel_mode: str, block_dim: int) -> Dict[str, Any]:
    changed = []
    tiling_header = project_dir / "op_host" / "fused_pressure_update_forward_tiling.h"
    kernel_cpp = project_dir / "op_kernel" / "fused_pressure_update_forward.cpp"
    host_cpp = project_dir / "op_host" / "fused_pressure_update_forward.cpp"

    tiling_header.write_text(TILING_HEADER)
    changed.append(str(tiling_header.relative_to(project_dir)))

    kernel_source = COPY_KERNEL_CPP if kernel_mode == "copy" else KERNEL_CPP
    kernel_cpp.write_text(kernel_source)
    changed.append(str(kernel_cpp.relative_to(project_dir)))

    host_text = host_cpp.read_text()
    if HOST_TILING_OLD not in host_text:
        raise RuntimeError("generated host tiling block did not match expected scaffold")
    host_tiling_new = HOST_TILING_NEW_TEMPLATE.replace("__BLOCK_DIM__", str(block_dim))
    host_cpp.write_text(host_text.replace(HOST_TILING_OLD, host_tiling_new))
    changed.append(str(host_cpp.relative_to(project_dir)))
    return {"changed_files": changed, "kernel_mode": kernel_mode, "block_dim": block_dim}


def run_build(project_dir: Path, cann_path: Path, timeout: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env["ASCEND_HOME_PATH"] = str(cann_path)
    try:
        completed = subprocess.run(
            ["bash", "build.sh"],
            cwd=str(project_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return {
            "command": ["bash", "build.sh"],
            "cwd": str(project_dir),
            "returncode": completed.returncode,
            "timed_out": False,
            "timeout_seconds": timeout,
            "output_tail": "\n".join(completed.stdout.splitlines()[-80:]),
        }
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        output = output or ""
        return {
            "command": ["bash", "build.sh"],
            "cwd": str(project_dir),
            "returncode": None,
            "timed_out": True,
            "timeout_seconds": timeout,
            "output_tail": "\n".join(output.splitlines()[-80:]),
        }


def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    workspace = Path(args.workspace).resolve() if args.workspace else Path(tempfile.mkdtemp(prefix="adfwi_pressure_update_op_"))
    workspace.mkdir(parents=True, exist_ok=True)
    workspace.chmod(0o700)
    ir_path = workspace / "fused_pressure_update_forward.json"
    project_dir = workspace / "out"
    write_ir_json(ir_path)

    generation = run_command(
        [
            str(Path(args.msopgen).resolve()),
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
        ],
        timeout=args.timeout,
    )

    status = "ok"
    patch_report = None
    python_patch_report = None
    compile_report = None
    generated_files = None
    if generation["returncode"] != 0:
        status = "generation_failed"
    else:
        python_patch_report = patch_generated_python(project_dir, Path(args.python_executable).resolve())
        patch_report = patch_pressure_update_project(
            project_dir,
            kernel_mode=args.kernel_mode,
            block_dim=args.block_dim,
        )
        generated_files = inspect_generated_files(project_dir)
        if args.compile:
            compile_report = run_build(project_dir, Path(args.cann_path).resolve(), args.compile_timeout)
            if compile_report.get("timed_out"):
                status = "compile_timed_out"
            elif compile_report["returncode"] != 0:
                status = "compile_failed"

    return {
        "status": status,
        "purpose": "minimal AscendC pressure-update custom-op compile gate",
        "workspace": str(workspace),
        "compile_requested": args.compile,
        "kernel_mode": args.kernel_mode,
        "block_dim": args.block_dim,
        "reference_contract": run_reference_contract(),
        "generation": generation,
        "patch_generated_python": python_patch_report,
        "patch_pressure_update_project": patch_report,
        "generated_files": generated_files,
        "compile": compile_report,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--msopgen", type=Path, default=DEFAULT_MSOPGEN)
    parser.add_argument("--cann-path", type=Path, default=DEFAULT_CANN)
    parser.add_argument("--compute-unit", default="ai_core-ascend910b2")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--compile-timeout", type=int, default=300)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--kernel-mode", choices=("pressure", "copy"), default="pressure")
    parser.add_argument("--block-dim", type=int, default=8)
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
                "compile_requested": report["compile_requested"],
                "kernel_mode": report["kernel_mode"],
                "block_dim": report["block_dim"],
                "compile_returncode": None if report["compile"] is None else report["compile"]["returncode"],
                "reference_contract": report["reference_contract"],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
