#!/usr/bin/env python
"""Profile acoustic timestep stencil components on the production shape.

This script does not modify or call the production propagator. It mirrors the
main ``step_forward`` tensor slices and measures the isolated cost of pressure,
source/free-surface, velocity, and receiver-sampling operations. The purpose is
to decide whether a formula-preserving stencil rewrite is worth attempting.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_timestep_stencil_breakdown_20260601.json"
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


def make_case(args: argparse.Namespace, *, seed_offset: int = 0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    device = torch.device(args.resolved_device)
    dtype = args.dtype
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc

    p = torch.randn((args.shots, nz_pml, nx_pml), device=device, dtype=dtype) * args.scale
    u = torch.randn((args.shots, nz_pml, nx_pml - 1), device=device, dtype=dtype) * args.scale
    w = torch.randn((args.shots, nz_pml - 1, nx_pml), device=device, dtype=dtype) * args.scale
    for tensor in (p, u, w):
        tensor.requires_grad_(True)

    damp = torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.01
    den = 1.0 + torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.1
    c = 2.0 + torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.1
    kappa1 = damp * args.dt
    alpha1 = den * c * c * args.dt / args.dz
    alpha2 = args.dt / (den * args.dz)
    kappa2 = torch.zeros_like(damp)
    kappa2[:, 1 : nx_pml - 2] = 0.5 * (damp[:, 1 : nx_pml - 2] + damp[:, 2 : nx_pml - 1]) * args.dt
    kappa3 = torch.zeros_like(damp)
    free_surface_start = args.nabc if args.free_surface else 1
    kappa3[free_surface_start : nz_pml - 2, :] = (
        0.5
        * (
            damp[free_surface_start : nz_pml - 2, :]
            + damp[free_surface_start + 1 : nz_pml - 1, :]
        )
        * args.dt
    )

    src_x = torch.linspace(args.nabc, args.nabc + args.nx - 1, args.shots, device=device).round().to(torch.long)
    src_z = torch.full((args.shots,), args.nabc + max(1, args.nz // 4), dtype=torch.long, device=device)
    src_index = torch.arange(args.shots, dtype=torch.long, device=device)
    src_v = torch.randn((args.shots,), device=device, dtype=dtype) * args.scale
    rcv_x = torch.linspace(args.nabc, args.nabc + args.nx - 1, args.receivers, device=device).round().to(torch.long)
    rcv_z = torch.full((args.receivers,), args.nabc + max(1, args.nz // 3), dtype=torch.long, device=device)

    return {
        "p": p,
        "u": u,
        "w": w,
        "kappa1": kappa1,
        "alpha1": alpha1,
        "kappa2": kappa2,
        "alpha2": alpha2,
        "kappa3": kappa3,
        "src_x": src_x,
        "src_z": src_z,
        "src_index": src_index,
        "src_v": src_v,
        "rcv_x": rcv_x,
        "rcv_z": rcv_z,
    }


def pressure_update(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    p_next = p.clone()
    p_next[:, zp, xp] = (
        (1.0 - kappa1[zp, xp]) * p[:, zp, xp]
        - alpha1[zp, xp]
        * (
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
    )
    return p_next


def source_free_surface(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    p_next = state["p"].clone()
    p_next[state["src_index"], state["src_z"], state["src_x"]] += args.dt * state["src_v"]
    if args.free_surface:
        p_next[:, free_surface_start - 1, :] = -p_next[:, free_surface_start + 1, :]
    return p_next


def horizontal_velocity_update(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p, u = state["p"], state["u"]
    kappa2, alpha2 = state["kappa2"], state["alpha2"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    zu = slice(free_surface_start, nz_pml - 1)
    xu = slice(1, nx_pml - 2)
    u_next = u.clone()
    u_next[:, zu, xu] = (
        (1.0 - kappa2[zu, xu]) * u[:, zu, xu]
        - alpha2[zu, xu]
        * (
            c1 * (p[:, zu, 2 : nx_pml - 1] - p[:, zu, 1 : nx_pml - 2])
            + c2 * (p[:, zu, 3:nx_pml] - p[:, zu, 0 : nx_pml - 3])
        )
    )
    return u_next


def vertical_velocity_update(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    p, w = state["p"], state["w"]
    kappa3, alpha2 = state["kappa3"], state["alpha2"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    zw = slice(free_surface_start, nz_pml - 2)
    xw = slice(1, nx_pml - 1)
    w_next = w.clone()
    w_next[:, zw, xw] = (
        (1.0 - kappa3[zw, xw]) * w[:, zw, xw]
        - alpha2[zw, xw]
        * (
            c1 * (p[:, free_surface_start + 1 : nz_pml - 1, xw] - p[:, zw, xw])
            + c2
            * (
                p[:, free_surface_start + 2 : nz_pml, xw]
                - p[:, free_surface_start - 1 : nz_pml - 3, xw]
            )
        )
    )
    if args.free_surface:
        w_next[:, free_surface_start - 1, :] = w_next[:, free_surface_start, :]
    return w_next


def receiver_sampling(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    rcv_x, rcv_z = state["rcv_x"], state["rcv_z"]
    return (
        state["p"][:, rcv_z, rcv_x].pow(2).mean()
        + state["u"][:, rcv_z, rcv_x].pow(2).mean()
        + state["w"][:, rcv_z, rcv_x].pow(2).mean()
    )


COMPONENTS: Dict[str, Callable[[argparse.Namespace, Dict[str, torch.Tensor]], torch.Tensor]] = {
    "pressure_update": pressure_update,
    "source_free_surface": source_free_surface,
    "horizontal_velocity_update": horizontal_velocity_update,
    "vertical_velocity_update": vertical_velocity_update,
    "receiver_sampling": receiver_sampling,
}


def run_component(args: argparse.Namespace, backend, name: str, *, seed_offset: int) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    timer = Timer(backend)
    fn = COMPONENTS[name]
    output, forward_seconds = timer.measure(lambda: fn(args, state))
    loss = output if output.ndim == 0 else output.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    grads = {key: state[key].grad for key in ("p", "u", "w")}
    return {
        "component": name,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": float(loss.detach().cpu().item()),
        "finite": {
            "output": bool(torch.isfinite(output).all().cpu().item()),
            "p_grad": grads["p"] is None or bool(torch.isfinite(grads["p"]).all().cpu().item()),
            "u_grad": grads["u"] is None or bool(torch.isfinite(grads["u"]).all().cpu().item()),
            "w_grad": grads["w"] is None or bool(torch.isfinite(grads["w"]).all().cpu().item()),
        },
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {"min": min(data), "max": max(data), "mean": sum(data) / len(data), "values": data}


def run_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device: Optional[str] = None if args.device == "auto" else args.device
    prefer: Tuple[str, ...] = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    backend = configure_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )
    args.resolved_device = backend.device

    components = [item.strip() for item in args.components.split(",") if item.strip()]
    unknown = sorted(set(components) - set(COMPONENTS))
    if unknown:
        raise ValueError(f"unknown component(s): {unknown}")

    for index in range(args.warmup):
        for name in components:
            run_component(args, backend, name, seed_offset=10000 + index)

    runs = {name: [] for name in components}
    for index in range(args.repeat):
        for name in components:
            runs[name].append(run_component(args, backend, name, seed_offset=index))

    summary = {}
    for name, items in runs.items():
        summary[name] = {
            "forward_seconds": summarize(item["forward_seconds"] for item in items),
            "backward_seconds": summarize(item["backward_seconds"] for item in items),
            "total_seconds": summarize(item["total_seconds"] for item in items),
            "all_finite": all(all(item["finite"].values()) for item in items),
        }

    return {
        "status": "ok",
        "purpose": "Acoustic timestep stencil component timing breakdown",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "receivers": args.receivers,
            "scale": args.scale,
            "dt": args.dt,
            "dz": args.dz,
            "free_surface": args.free_surface,
            "components": components,
        },
        "summary": summary,
        "runs": runs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--nz", type=int, default=88)
    parser.add_argument("--nabc", type=int, default=30)
    parser.add_argument("--receivers", type=int, default=200)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--dz", type=float, default=40.0)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--components",
        default="pressure_update,source_free_surface,horizontal_velocity_update,vertical_velocity_update,receiver_sampling",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
