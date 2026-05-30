#!/usr/bin/env python
"""Measure acoustic receiver-sampling and forward-wavefield output costs.

This is a micro-benchmark for the output-side operations inside
``ADFWI.propagator.acoustic_kernels.step_forward``. It intentionally does not
replace or modify the production propagator. The benchmark repeats the same
tensor expressions used by the kernel:

- receiver sampling: ``p[:, rcv_z, rcv_x]`` and optionally ``u/w``;
- forward-wavefield accumulation:
  ``torch.sum(p * p, dim=0)[nabc:nabc+nz, nabc:nabc+nx].detach()`` and
  optionally ``u/w``.

The goal is to estimate whether output recording/accumulation is large enough
to justify a real kernel-level optimization. It is not a numerical replacement
for full acoustic propagation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import BackendUnavailableError, configure_backend
from scripts.smoke.acoustic_backend_smoke import parse_dtype


MODE_FLAGS = {
    "empty": (False, False, False, False),
    "receiver_p": (True, False, False, False),
    "receiver_puw": (True, True, False, False),
    "wavefield_p": (False, False, True, False),
    "wavefield_puw": (False, False, True, True),
    "receiver_puw_wavefield_puw": (True, True, True, True),
}


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(sum(data) / len(data)),
    }


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "device": str(detached.device),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(detached).all().cpu().item()),
        "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
    }


def build_state(args: argparse.Namespace, backend) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    nx_pml = args.nx + 2 * args.nabc
    nz_pml = args.nz + 2 * args.nabc

    p = torch.randn(
        (args.shots, nz_pml, nx_pml),
        generator=generator,
        dtype=args.dtype,
        device="cpu",
    ).to(backend.device)
    u = torch.randn(
        (args.shots, nz_pml, nx_pml - 1),
        generator=generator,
        dtype=args.dtype,
        device="cpu",
    ).to(backend.device)
    w = torch.randn(
        (args.shots, nz_pml - 1, nx_pml),
        generator=generator,
        dtype=args.dtype,
        device="cpu",
    ).to(backend.device)

    rcv_x_np = np.linspace(args.nabc, args.nabc + args.nx - 1, args.receivers, dtype=np.int64)
    rcv_z_np = np.full(args.receivers, args.nabc + max(1, args.nz // 4), dtype=np.int64)
    rcv_x = torch.as_tensor(rcv_x_np, dtype=torch.long, device=backend.device)
    rcv_z = torch.as_tensor(rcv_z_np, dtype=torch.long, device=backend.device)

    if args.requires_grad:
        p.requires_grad_(True)
        u.requires_grad_(True)
        w.requires_grad_(True)

    return {"p": p, "u": u, "w": w, "rcv_x": rcv_x, "rcv_z": rcv_z}


def run_output_loop(
    args: argparse.Namespace,
    state: Dict[str, torch.Tensor],
    *,
    record_receivers: bool,
    record_velocity: bool,
    accumulate_wavefield: bool,
    accumulate_velocity: bool,
) -> Dict[str, torch.Tensor]:
    p = state["p"]
    u = state["u"]
    w = state["w"]
    rcv_x = state["rcv_x"]
    rcv_z = state["rcv_z"]

    rcv_p = torch.zeros((args.shots, args.nt, args.receivers), dtype=args.dtype, device=p.device)
    rcv_u = torch.zeros_like(rcv_p)
    rcv_w = torch.zeros_like(rcv_p)
    wf_p = torch.zeros((args.nz, args.nx), dtype=args.dtype, device=p.device)
    wf_u = torch.zeros_like(wf_p)
    wf_w = torch.zeros_like(wf_p)

    z_slice = slice(args.nabc, args.nabc + args.nz)
    x_slice = slice(args.nabc, args.nabc + args.nx)

    for it in range(args.nt):
        if record_receivers:
            rcv_p[:, it, :] = p[:, rcv_z, rcv_x]
            if record_velocity:
                rcv_u[:, it, :] = u[:, rcv_z, rcv_x]
                rcv_w[:, it, :] = w[:, rcv_z, rcv_x]

        if accumulate_wavefield:
            wf_p = wf_p + torch.sum(p * p, dim=0)[z_slice, x_slice].detach()
            if accumulate_velocity:
                wf_u = wf_u + torch.sum(u * u, dim=0)[z_slice, x_slice].detach()
                wf_w = wf_w + torch.sum(w * w, dim=0)[z_slice, x_slice].detach()

    return {
        "p": rcv_p,
        "u": rcv_u,
        "w": rcv_w,
        "forward_wavefield_p": wf_p,
        "forward_wavefield_u": wf_u,
        "forward_wavefield_w": wf_w,
    }


def run_mode(args: argparse.Namespace, backend, mode: str) -> Dict[str, Any]:
    flags = MODE_FLAGS[mode]
    state = build_state(args, backend)

    synchronize(backend)
    start = time.perf_counter()
    outputs = run_output_loop(
        args,
        state,
        record_receivers=flags[0],
        record_velocity=flags[1],
        accumulate_wavefield=flags[2],
        accumulate_velocity=flags[3],
    )
    synchronize(backend)
    forward_seconds = time.perf_counter() - start

    loss = outputs["p"].pow(2).mean()
    backward_seconds: Optional[float] = None
    grad_summaries: Optional[Dict[str, Dict[str, Any]]] = None

    if args.requires_grad and flags[0]:
        synchronize(backend)
        start = time.perf_counter()
        loss.backward()
        synchronize(backend)
        backward_seconds = time.perf_counter() - start
        grad_summaries = {
            key: tensor_summary(state[key].grad)
            for key in ("p", "u", "w")
            if state[key].grad is not None
        }

    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + (backward_seconds or 0.0),
        "loss": float(loss.detach().cpu().item()),
        "outputs": {
            key: tensor_summary(value)
            for key, value in outputs.items()
        },
        "grad_summaries": grad_summaries,
        "memory_allocated": backend.memory_allocated(),
    }


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(MODE_FLAGS))
    if unknown:
        raise ValueError(f"unsupported modes: {unknown}; expected one of {sorted(MODE_FLAGS)}")

    warmups = []
    for _ in range(args.warmup):
        for mode in modes:
            warmups.append(run_mode(args, backend, mode))

    runs: Dict[str, list[Dict[str, Any]]] = {mode: [] for mode in modes}
    for _ in range(args.repeat):
        for mode in modes:
            runs[mode].append(run_mode(args, backend, mode))

    summary = {
        mode: {
            "forward_seconds": summarize(run["forward_seconds"] for run in mode_runs),
            "backward_seconds": summarize(
                run["backward_seconds"] for run in mode_runs if run["backward_seconds"] is not None
            )
            if any(run["backward_seconds"] is not None for run in mode_runs)
            else None,
            "total_seconds": summarize(run["total_seconds"] for run in mode_runs),
        }
        for mode, mode_runs in runs.items()
    }

    return {
        "status": "ok",
        "command": [sys.executable, *sys.argv],
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "fallback_cpu": args.fallback_cpu,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "modes": modes,
            "requires_grad": args.requires_grad,
            "shots": args.shots,
            "receivers": args.receivers,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
        },
        "summary": summary,
        "runs": runs,
        "warmups": warmups if args.include_warmup else [],
    }


def write_report(report: Dict[str, Any], output: Optional[str]) -> None:
    if output is None:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype, help="float32 or float64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument(
        "--modes",
        default="empty,receiver_p,receiver_puw,wavefield_p,wavefield_puw,receiver_puw_wavefield_puw",
        help="comma-separated modes",
    )
    parser.add_argument("--requires-grad", action="store_true", help="also time backward for receiver modes")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--receivers", type=int, default=200)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--output", help="optional JSON report path")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    try:
        report = run_experiment(args)
    except BackendUnavailableError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1

    write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
