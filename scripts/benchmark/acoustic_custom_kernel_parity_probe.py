#!/usr/bin/env python
"""Compare production acoustic forward_kernel with a custom-gradient prototype.

This is an experimental parity harness. It does not change production
propagator code. The purpose is to test whether the isolated custom-gradient
recurrence can match the public acoustic kernel receiver outputs and raw
velocity gradient on a tiny case.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from ADFWI.propagator.acoustic_kernels import forward_kernel
from scripts.benchmark.acoustic_custom_multistep_update_probe import tensor_diff
from scripts.benchmark.acoustic_experimental_forward import experimental_forward_kernel
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_custom_kernel_parity_probe_20260531.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


class Timer:
    def __init__(self, backend) -> None:
        self.backend = backend

    def measure(self, fn):
        synchronize(self.backend)
        start = time.perf_counter()
        value = fn()
        synchronize(self.backend)
        return value, time.perf_counter() - start


def build_case(args: argparse.Namespace, *, device: torch.device, dtype: torch.dtype, seed: int):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    v_cpu = args.velocity_base + args.velocity_scale * torch.rand(args.nz, args.nx, generator=generator, dtype=dtype)
    rho_cpu = torch.ones(args.nz, args.nx, dtype=dtype) * args.rho
    damp_cpu = torch.zeros(args.nz + 2 * args.nabc, args.nx + 2 * args.nabc, dtype=dtype)
    wavelet_cpu = args.source_scale * torch.sin(torch.arange(args.nt, dtype=dtype) * 0.17).reshape(1, args.nt)
    rcv_x_cpu = torch.linspace(0, args.nx - 1, args.receivers, dtype=dtype).to(torch.long)
    rcv_z_cpu = torch.full((args.receivers,), args.receiver_depth, dtype=torch.long)
    src_x_cpu = torch.tensor([args.nx // 2], dtype=torch.long)
    src_z_cpu = torch.tensor([args.source_depth], dtype=torch.long)
    return {
        "v": v_cpu.to(device=device).requires_grad_(True),
        "rho": rho_cpu.to(device=device),
        "damp": damp_cpu.to(device=device),
        "src_x": src_x_cpu.to(device=device),
        "src_z": src_z_cpu.to(device=device),
        "src_v": wavelet_cpu.to(device=device),
        "rcv_x": rcv_x_cpu.to(device=device),
        "rcv_z": rcv_z_cpu.to(device=device),
    }


def run_production(case: Dict[str, torch.Tensor], args: argparse.Namespace, backend) -> Dict[str, Any]:
    timer = Timer(backend)

    def forward():
        return forward_kernel(
            args.nx,
            args.nz,
            args.dx,
            args.dz,
            args.nt,
            args.dt,
            args.nabc,
            args.free_surface,
            case["src_x"],
            case["src_z"],
            1,
            case["src_v"],
            case["rcv_x"],
            case["rcv_z"],
            args.receivers,
            case["damp"],
            case["v"],
            case["rho"],
            checkpoint_segments=1,
            save_forward_wavefield=False,
            device=backend.device,
            dtype=backend.dtype,
        )

    record, forward_seconds = timer.measure(forward)
    outputs = (record["p"], record["u"], record["w"])
    loss = sum(output.pow(2).mean() for output in outputs)
    _, backward_seconds = timer.measure(lambda: loss.backward())
    if case["v"].grad is None:
        raise RuntimeError("production v.grad is None")
    return {
        "outputs": [output.detach() for output in outputs],
        "loss": float(loss.detach().cpu().item()),
        "v_grad": case["v"].grad.detach().clone(),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
    }


def run_custom(case: Dict[str, torch.Tensor], args: argparse.Namespace, backend) -> Dict[str, Any]:
    timer = Timer(backend)

    def forward():
        record = experimental_forward_kernel(
            args.nx,
            args.nz,
            args.dx,
            args.dz,
            args.nt,
            args.dt,
            args.nabc,
            args.free_surface,
            case["src_x"],
            case["src_z"],
            1,
            case["src_v"],
            case["rcv_x"],
            case["rcv_z"],
            args.receivers,
            case["damp"],
            case["v"],
            case["rho"],
            save_forward_wavefield=False,
            device=backend.device,
            dtype=backend.dtype,
        )
        return record["p"], record["u"], record["w"]

    outputs, forward_seconds = timer.measure(forward)
    loss = sum(output.pow(2).mean() for output in outputs)
    _, backward_seconds = timer.measure(lambda: loss.backward())
    if case["v"].grad is None:
        raise RuntimeError("custom v.grad is None")
    return {
        "outputs": [output.detach() for output in outputs],
        "loss": float(loss.detach().cpu().item()),
        "v_grad": case["v"].grad.detach().clone(),
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
    }


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss_abs_diff": abs(candidate["loss"] - reference["loss"]),
        "outputs": {
            name: tensor_diff(ref, cand)
            for name, ref, cand in zip(("rcv_p", "rcv_u", "rcv_w"), reference["outputs"], candidate["outputs"])
        },
        "v_grad": tensor_diff(reference["v_grad"], candidate["v_grad"]),
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize(values: Iterable[float]) -> Dict[str, Any]:
    data = list(values)
    return {"min": float(min(data)), "max": float(max(data)), "mean": float(sum(data) / len(data)), "values": data}


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(device=requested_device, dtype=parse_dtype(args.dtype), prefer=prefer)
    pairs = []
    for index in range(args.warmup):
        seed = args.seed + 10000 + index
        run_production(build_case(args, device=backend.device, dtype=backend.dtype, seed=seed), args, backend)
        run_custom(build_case(args, device=backend.device, dtype=backend.dtype, seed=seed), args, backend)
    for index in range(args.repeat):
        seed = args.seed + index
        reference = run_production(build_case(args, device=backend.device, dtype=backend.dtype, seed=seed), args, backend)
        candidate = run_custom(build_case(args, device=backend.device, dtype=backend.dtype, seed=seed), args, backend)
        pairs.append({"reference": _public_run(reference), "candidate": _public_run(candidate), "comparison": compare_pair(reference, candidate)})
    return {
        "backend": backend.diagnostics(),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": {
            "speedup": {
                "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pairs),
                "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pairs),
                "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pairs),
            },
            "max_differences": {
                "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
                "output_max_abs_diff": max(
                    diff["max_abs_diff"] for pair in pairs for diff in pair["comparison"]["outputs"].values()
                ),
                "output_max_rel_diff": max(
                    diff["max_rel_diff"] for pair in pairs for diff in pair["comparison"]["outputs"].values()
                ),
                "v_grad_max_abs_diff": max(pair["comparison"]["v_grad"]["max_abs_diff"] for pair in pairs),
                "v_grad_max_rel_diff": max(pair["comparison"]["v_grad"]["max_rel_diff"] for pair in pairs),
            },
        },
        "pairs": pairs,
    }


def _public_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss": run["loss"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prefer", default="npu,cuda,cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=20240531)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--nabc", type=int, default=8)
    parser.add_argument("--nt", type=int, default=20)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--receivers", type=int, default=16)
    parser.add_argument("--receiver-depth", type=int, default=1)
    parser.add_argument("--source-depth", type=int, default=1)
    parser.add_argument("--velocity-base", type=float, default=1500.0)
    parser.add_argument("--velocity-scale", type=float, default=1200.0)
    parser.add_argument("--rho", type=float, default=1000.0)
    parser.add_argument("--source-scale", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.nt <= 0:
        parser.error("--nt must be positive")
    if args.receivers <= 0:
        parser.error("--receivers must be positive")
    if args.receivers > args.nx:
        parser.error("--receivers must be <= --nx for this parity probe")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
