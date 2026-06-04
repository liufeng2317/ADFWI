#!/usr/bin/env python
"""Probe an inner-state acoustic pressure recurrence.

This bounded Phase B benchmark tests whether the recurrent pressure update
itself benefits from avoiding full-field sliced assignment. It compares:

- reference: clone full pressure field and update the interior slice each step;
- candidate: keep only the pressure interior as the recurrent state.

The benchmark does not modify the production propagator. It checks output and
gradients for ``p``, ``u_seq``, ``w_seq``, ``kappa1``, and ``alpha1``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from scripts.benchmark.acoustic_timestep_update_microbenchmark import Timer, summarize, tensor_diff
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_pressure_inner_state_microbenchmark_20260531.json"
)


def make_case(args: argparse.Namespace, *, seed_offset: int = 0) -> Dict[str, torch.Tensor]:
    torch.manual_seed(args.seed + seed_offset)
    np.random.seed(args.seed + seed_offset)
    device = torch.device(args.resolved_device)
    dtype = args.dtype
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc

    p = torch.randn((args.shots, nz_pml, nx_pml), device=device, dtype=dtype) * args.scale
    u_seq = torch.randn((args.nt, args.shots, nz_pml, nx_pml - 1), device=device, dtype=dtype) * args.scale
    w_seq = torch.randn((args.nt, args.shots, nz_pml - 1, nx_pml), device=device, dtype=dtype) * args.scale
    kappa1 = torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.01
    alpha1 = torch.rand((nz_pml, nx_pml), device=device, dtype=dtype) * 0.02
    for tensor in (p, u_seq, w_seq, kappa1, alpha1):
        tensor.requires_grad_(True)
    return {"p": p, "u_seq": u_seq, "w_seq": w_seq, "kappa1": kappa1, "alpha1": alpha1}


def pressure_update(
    args: argparse.Namespace,
    p_inner: torch.Tensor,
    u: torch.Tensor,
    w: torch.Tensor,
    kappa_inner: torch.Tensor,
    alpha_inner: torch.Tensor,
) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    return (
        (1.0 - kappa_inner) * p_inner
        - alpha_inner
        * (
            c1
            * (
                u[:, z0:z1, x0:x1]
                - u[:, z0:z1, x0 - 1:x1 - 1]
                + w[:, z0:z1, x0:x1]
                - w[:, z0 - 1:z1 - 1, x0:x1]
            )
            + c2
            * (
                u[:, z0:z1, x0 + 1:x1 + 1]
                - u[:, z0:z1, x0 - 2:x1 - 2]
                + w[:, z0 + 1:z1 + 1, x0:x1]
                - w[:, z0 - 2:z1 - 2, x0:x1]
            )
        )
    )


def run_reference(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    p = state["p"].clone()
    kappa_inner = state["kappa1"][z0:z1, x0:x1]
    alpha_inner = state["alpha1"][z0:z1, x0:x1]
    for it in range(args.nt):
        # Clone the RHS view before the sliced assignment. The production
        # TorchScript/checkpoint path can replay its in-place state updates, but
        # this standalone Python autograd probe needs an unmodified RHS tensor
        # to make the reference branch a valid gradient baseline.
        p[:, z0:z1, x0:x1] = pressure_update(
            args,
            p[:, z0:z1, x0:x1].clone(),
            state["u_seq"][it],
            state["w_seq"][it],
            kappa_inner,
            alpha_inner,
        )
    return p[:, z0:z1, x0:x1]


def run_candidate(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    p_inner = state["p"][:, z0:z1, x0:x1]
    kappa_inner = state["kappa1"][z0:z1, x0:x1]
    alpha_inner = state["alpha1"][z0:z1, x0:x1]
    for it in range(args.nt):
        p_inner = pressure_update(args, p_inner, state["u_seq"][it], state["w_seq"][it], kappa_inner, alpha_inner)
    return p_inner


def run_variant(args: argparse.Namespace, backend, mode: str, *, seed_offset: int) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    fn = run_reference if mode == "reference" else run_candidate
    timer = Timer(backend)
    output, forward_seconds = timer.measure(lambda: fn(args, state))
    loss = output.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    gradients = {name: state[name].grad.detach().clone() for name in ("p", "u_seq", "w_seq", "kappa1", "alpha1")}
    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "output": output.detach().clone(),
        "gradients": gradients,
        "finite": {
            "output": bool(torch.isfinite(output).all().cpu().item()),
            **{
                f"{name}_grad": bool(torch.isfinite(grad).all().cpu().item())
                for name, grad in gradients.items()
            },
        },
    }


def public_variant(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mode": run["mode"],
        "forward_seconds": run["forward_seconds"],
        "backward_seconds": run["backward_seconds"],
        "total_seconds": run["total_seconds"],
        "loss": float(run["loss"].cpu().item()),
        "finite": run["finite"],
    }


def compare_pair(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "loss_abs_diff": float((candidate["loss"].cpu() - reference["loss"].cpu()).abs().item()),
        "output": tensor_diff(reference["output"], candidate["output"]),
        "gradients": {
            name: tensor_diff(reference["gradients"][name], candidate["gradients"][name])
            for name in ("p", "u_seq", "w_seq", "kappa1", "alpha1")
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
        },
    }


def summarize_pairs(pairs: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    pair_list = list(pairs)
    return {
        "speedup": {
            "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pair_list),
            "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pair_list),
            "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pair_list),
        },
        "max_differences": {
            "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pair_list),
            "output_max_abs_diff": max(pair["comparison"]["output"]["max_abs_diff"] for pair in pair_list),
            **{
                f"{name}_grad_max_abs_diff": max(pair["comparison"]["gradients"][name]["max_abs_diff"] for pair in pair_list)
                for name in ("p", "u_seq", "w_seq", "kappa1", "alpha1")
            },
        },
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
    args.resolved_device = backend.device

    warmups = []
    for index in range(args.warmup):
        try:
            reference = run_variant(args, backend, "reference", seed_offset=10000 + index)
            candidate = run_variant(args, backend, "candidate", seed_offset=10000 + index)
        except Exception as exc:  # pragma: no cover - backend/autograd dependent
            return failure_report(args, backend, "warmup", index, exc)
        warmups.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    pairs = []
    for index in range(args.repeat):
        try:
            reference = run_variant(args, backend, "reference", seed_offset=index)
            candidate = run_variant(args, backend, "candidate", seed_offset=index)
        except Exception as exc:  # pragma: no cover - backend/autograd dependent
            return failure_report(args, backend, "repeat", index, exc, warmups=warmups)
        pairs.append(
            {
                "reference": public_variant(reference),
                "candidate": public_variant(candidate),
                "comparison": compare_pair(reference, candidate),
            }
        )

    return {
        "status": "ok",
        "purpose": "Phase B acoustic pressure inner-state recurrence probe",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "scale": args.scale,
            "free_surface": args.free_surface,
        },
        "warmups": warmups if args.include_warmup else [],
        "summary": summarize_pairs(pairs),
        "pairs": pairs,
    }


def failure_report(
    args: argparse.Namespace,
    backend,
    stage: str,
    index: int,
    exc: Exception,
    *,
    warmups=None,
) -> Dict[str, Any]:
    return {
        "status": "failed",
        "purpose": "Phase B acoustic pressure inner-state recurrence probe",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "scale": args.scale,
            "free_surface": args.free_surface,
        },
        "failure": {
            "stage": stage,
            "index": index,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=12),
        },
        "warmups": warmups or [],
        "summary": None,
        "pairs": [],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--include-warmup", action="store_true")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
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
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
