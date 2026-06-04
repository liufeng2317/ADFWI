#!/usr/bin/env python
"""Compare acoustic pressure operator backends on a small differentiable case."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from ADFWI.propagator.acoustic_operator import (
    AcousticOperatorConfig,
    AcousticOperatorInputs,
    acoustic_pressure_operator,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_operator_backend_compare_20260604.json"
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize(device)


def memory_api(device: torch.device):
    if device.type == "cuda":
        return torch.cuda
    if device.type == "npu" and hasattr(torch, "npu"):
        return torch.npu
    return None


def reset_peak_memory(device: torch.device) -> None:
    api = memory_api(device)
    if api is None:
        return
    if hasattr(api, "empty_cache"):
        api.empty_cache()
    if hasattr(api, "reset_peak_memory_stats"):
        api.reset_peak_memory_stats(device)
    elif hasattr(api, "reset_max_memory_allocated"):
        api.reset_max_memory_allocated(device)


def max_memory_allocated(device: torch.device) -> Optional[int]:
    api = memory_api(device)
    if api is None or not hasattr(api, "max_memory_allocated"):
        return None
    try:
        return int(api.max_memory_allocated(device))
    except TypeError:
        return int(api.max_memory_allocated())


def bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def make_case(args: argparse.Namespace, device: torch.device):
    dtype = getattr(torch, args.dtype)
    config = AcousticOperatorConfig(
        nx=args.nx,
        nz=args.nz,
        dx=args.dx,
        dz=args.dz,
        nt=args.nt,
        dt=args.dt,
        nabc=args.nabc,
        free_surface=args.free_surface,
        checkpoint_segments=args.checkpoint_segments,
    )

    src_x = torch.linspace(2, args.nx - 3, args.shots, device=device).round().long()
    src_z = torch.full((args.shots,), 2, dtype=torch.long, device=device)
    rcv_x = torch.linspace(1, args.nx - 2, args.receivers, device=device).round().long()
    rcv_z = torch.full((args.receivers,), 2, dtype=torch.long, device=device)

    src_v = torch.zeros((args.shots, args.nt), dtype=dtype, device=device)
    source_time = min(max(args.source_time, 0), args.nt - 1)
    src_v[:, source_time] = args.source_amplitude

    damp = torch.zeros(
        (args.nz + 2 * args.nabc, args.nx + 2 * args.nabc),
        dtype=dtype,
        device=device,
    )
    rho = torch.full((args.nz, args.nx), args.rho, dtype=dtype, device=device)
    vp_template = torch.full((args.nz, args.nx), args.vp, dtype=dtype, device=device)

    return config, dict(
        src_x=src_x,
        src_z=src_z,
        src_v=src_v,
        rcv_x=rcv_x,
        rcv_z=rcv_z,
        damp=damp,
        rho=rho,
        vp_template=vp_template,
    )


def run_backend(
    backend: str,
    config: AcousticOperatorConfig,
    tensors: Dict[str, torch.Tensor],
    device: torch.device,
):
    vp = tensors["vp_template"].detach().clone().requires_grad_(True)
    inputs = AcousticOperatorInputs(
        src_x=tensors["src_x"],
        src_z=tensors["src_z"],
        src_v=tensors["src_v"],
        rcv_x=tensors["rcv_x"],
        rcv_z=tensors["rcv_z"],
        damp=tensors["damp"],
        vp=vp,
        rho=tensors["rho"],
    )

    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()
    output = acoustic_pressure_operator(config, inputs, backend=backend)
    loss = output.square().sum()
    loss.backward()
    synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "elapsed_s": elapsed,
        "peak_memory_bytes": max_memory_allocated(device),
        "output": output.detach(),
        "loss": loss.detach(),
        "vp_grad": vp.grad.detach(),
    }


def summarize_runs(runs):
    elapsed = [run["elapsed_s"] for run in runs]
    memory = [run["peak_memory_bytes"] for run in runs if run["peak_memory_bytes"] is not None]
    return {
        "elapsed_s_all": elapsed,
        "elapsed_s_median": statistics.median(elapsed),
        "elapsed_s_mean": statistics.mean(elapsed),
        "peak_memory_mib_all": [bytes_to_mib(value) for value in memory],
        "peak_memory_mib_max": bytes_to_mib(max(memory)) if memory else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--nt", type=int, default=80)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nabc", type=int, default=4)
    parser.add_argument("--shots", type=int, default=2)
    parser.add_argument("--receivers", type=int, default=16)
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--source-time", type=int, default=1)
    parser.add_argument("--source-amplitude", type=float, default=1.0)
    parser.add_argument("--vp", type=float, default=2000.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--free-surface", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    device = torch.device(args.device)
    config, tensors = make_case(args, device)

    backends = ["torch_reference", "custom_autograd_remat"]
    for _ in range(args.warmups):
        for backend in backends:
            run_backend(backend, config, tensors, device)

    runs = {backend: [] for backend in backends}
    for _ in range(args.runs):
        for backend in backends:
            runs[backend].append(run_backend(backend, config, tensors, device))

    reference = runs["torch_reference"][-1]
    remat = runs["custom_autograd_remat"][-1]
    output_diff = (reference["output"] - remat["output"]).abs()
    grad_diff = (reference["vp_grad"] - remat["vp_grad"]).abs()
    ref_grad_norm = reference["vp_grad"].abs().max().item()

    ref_summary = summarize_runs(runs["torch_reference"])
    remat_summary = summarize_runs(runs["custom_autograd_remat"])
    result: Dict[str, Any] = {
        "case": {
            "device": args.device,
            "dtype": args.dtype,
            "nx": args.nx,
            "nz": args.nz,
            "nt": args.nt,
            "shots": args.shots,
            "receivers": args.receivers,
            "checkpoint_segments": args.checkpoint_segments,
            "warmups": args.warmups,
            "runs": args.runs,
        },
        "torch_reference": ref_summary,
        "custom_autograd_remat": remat_summary,
        "speedup_reference_over_remat": (
            ref_summary["elapsed_s_median"] / remat_summary["elapsed_s_median"]
        ),
        "memory_ratio_remat_over_reference": (
            remat_summary["peak_memory_mib_max"] / ref_summary["peak_memory_mib_max"]
            if remat_summary["peak_memory_mib_max"] is not None
            and ref_summary["peak_memory_mib_max"] not in (None, 0)
            else None
        ),
        "parity": {
            "output_max_abs_diff": output_diff.max().item(),
            "vp_grad_max_abs_diff": grad_diff.max().item(),
            "vp_grad_reference_max_abs": ref_grad_norm,
            "vp_grad_relative_diff": (
                grad_diff.max().item() / ref_grad_norm if ref_grad_norm else 0.0
            ),
            "reference_vp_grad_finite": bool(torch.isfinite(reference["vp_grad"]).all().item()),
            "remat_vp_grad_finite": bool(torch.isfinite(remat["vp_grad"]).all().item()),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
