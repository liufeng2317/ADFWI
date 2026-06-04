#!/usr/bin/env python
"""Compare formula-equivalent acoustic pressure update variants.

This benchmark does not modify the production propagator. It isolates the
pressure stencil, compares expression variants against the production-style
sliced assignment, and reports output/gradient parity plus timing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from scripts.benchmark.acoustic_timestep_stencil_breakdown import Timer, make_case, synchronize
from ADFWI.backends import configure_backend
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_pressure_update_variants_20260601.json"
)


def pressure_slices(args: argparse.Namespace):
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    zp = slice(free_surface_start + 1, nz_pml - 2)
    xp = slice(2, nx_pml - 2)
    return free_surface_start, nz_pml, nx_pml, zp, xp


def update_reference(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start, nz_pml, nx_pml, zp, xp = pressure_slices(args)
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
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


def update_explicit_div(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start, nz_pml, nx_pml, zp, xp = pressure_slices(args)
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
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
    p_next = p.clone()
    p_next[:, zp, xp] = (1.0 - kappa1[zp, xp]) * p[:, zp, xp] - alpha1[zp, xp] * div_p
    return p_next


def update_split_div(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start, nz_pml, nx_pml, zp, xp = pressure_slices(args)
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
    div_u = c1 * (u[:, zp, 2 : nx_pml - 2] - u[:, zp, 1 : nx_pml - 3]) + c2 * (
        u[:, zp, 3 : nx_pml - 1] - u[:, zp, 0 : nx_pml - 4]
    )
    div_w = c1 * (
        w[:, zp, 2 : nx_pml - 2] - w[:, free_surface_start : nz_pml - 3, 2 : nx_pml - 2]
    ) + c2 * (
        w[:, free_surface_start + 2 : nz_pml - 1, 2 : nx_pml - 2]
        - w[:, free_surface_start - 1 : nz_pml - 4, 2 : nx_pml - 2]
    )
    p_next = p.clone()
    p_next[:, zp, xp] = (1.0 - kappa1[zp, xp]) * p[:, zp, xp] - alpha1[zp, xp] * (div_u + div_w)
    return p_next


def update_addcmul(args: argparse.Namespace, state: Dict[str, torch.Tensor]) -> torch.Tensor:
    free_surface_start, nz_pml, nx_pml, zp, xp = pressure_slices(args)
    p, u, w = state["p"], state["u"], state["w"]
    kappa1, alpha1 = state["kappa1"], state["alpha1"]
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0
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
    p_inner = p[:, zp, xp]
    updated = torch.addcmul(p_inner, kappa1[zp, xp], p_inner, value=-1.0)
    updated = torch.addcmul(updated, alpha1[zp, xp], div_p, value=-1.0)
    p_next = p.clone()
    p_next[:, zp, xp] = updated
    return p_next


VARIANTS: Dict[str, Callable[[argparse.Namespace, Dict[str, torch.Tensor]], torch.Tensor]] = {
    "reference": update_reference,
    "explicit_div": update_explicit_div,
    "split_div": update_split_div,
    "addcmul": update_addcmul,
}


def tensor_diff(reference: torch.Tensor, candidate: torch.Tensor, *, atol_floor=1e-12) -> Dict[str, Any]:
    ref = reference.detach().cpu()
    val = candidate.detach().cpu()
    diff = (val - ref).abs()
    denom = torch.maximum(ref.abs(), torch.full_like(ref, atol_floor))
    rel = diff / denom
    return {
        "shape": list(ref.shape),
        "max_abs_diff": float(diff.max().item()),
        "max_rel_diff": float(rel.max().item()),
        "reference_norm": float(torch.linalg.norm(ref.reshape(-1)).item()),
        "candidate_norm": float(torch.linalg.norm(val.reshape(-1)).item()),
    }


def run_variant(args: argparse.Namespace, backend, variant: str, *, seed_offset: int) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    timer = Timer(backend)
    output, forward_seconds = timer.measure(lambda: VARIANTS[variant](args, state))
    loss = output.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    gradients = {name: state[name].grad.detach().clone() for name in ("p", "u", "w")}
    return {
        "variant": variant,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "output": output.detach().clone(),
        "gradients": gradients,
        "finite": {
            "output": bool(torch.isfinite(output).all().cpu().item()),
            "p_grad": bool(torch.isfinite(gradients["p"]).all().cpu().item()),
            "u_grad": bool(torch.isfinite(gradients["u"]).all().cpu().item()),
            "w_grad": bool(torch.isfinite(gradients["w"]).all().cpu().item()),
        },
    }


def public_variant(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "variant": run["variant"],
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
            for name in ("p", "u", "w")
        },
        "speedup": {
            "forward": reference["forward_seconds"] / candidate["forward_seconds"],
            "backward": reference["backward_seconds"] / candidate["backward_seconds"],
            "total": reference["total_seconds"] / candidate["total_seconds"],
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
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown:
        raise ValueError(f"unknown variant(s): {unknown}")
    if "reference" in variants:
        variants.remove("reference")

    for index in range(args.warmup):
        reference = run_variant(args, backend, "reference", seed_offset=10000 + index)
        for variant in variants:
            run_variant(args, backend, variant, seed_offset=10000 + index)

    pairs = {variant: [] for variant in variants}
    for index in range(args.repeat):
        reference = run_variant(args, backend, "reference", seed_offset=index)
        for variant in variants:
            candidate = run_variant(args, backend, variant, seed_offset=index)
            pairs[variant].append(
                {
                    "reference": public_variant(reference),
                    "candidate": public_variant(candidate),
                    "comparison": compare_pair(reference, candidate),
                }
            )

    summary = {}
    for variant, items in pairs.items():
        summary[variant] = {
            "speedup": {
                "forward": summarize(item["comparison"]["speedup"]["forward"] for item in items),
                "backward": summarize(item["comparison"]["speedup"]["backward"] for item in items),
                "total": summarize(item["comparison"]["speedup"]["total"] for item in items),
            },
            "max_differences": {
                "loss_abs_diff": max(item["comparison"]["loss_abs_diff"] for item in items),
                "output_max_abs_diff": max(item["comparison"]["output"]["max_abs_diff"] for item in items),
                "p_grad_max_abs_diff": max(item["comparison"]["gradients"]["p"]["max_abs_diff"] for item in items),
                "u_grad_max_abs_diff": max(item["comparison"]["gradients"]["u"]["max_abs_diff"] for item in items),
                "w_grad_max_abs_diff": max(item["comparison"]["gradients"]["w"]["max_abs_diff"] for item in items),
            },
        }

    return {
        "status": "ok",
        "purpose": "Acoustic pressure update formula-equivalent variant benchmark",
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
            "scale": args.scale,
            "dt": args.dt,
            "dz": args.dz,
            "free_surface": args.free_surface,
            "variants": variants,
        },
        "summary": summary,
        "pairs": pairs,
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
    parser.add_argument("--variants", default="explicit_div,split_div,addcmul")
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
