#!/usr/bin/env python
"""Probe torch.compile feasibility for the acoustic timestep update.

This benchmark is intentionally isolated from the production propagator. It
checks whether graph capture can compile a representative acoustic pressure
update on the active backend while preserving autograd outputs and gradients.
The result decides whether ``torch.compile`` deserves a real propagator-level
experiment or should remain a research-only option for the current environment.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from ADFWI.backends import configure_backend
from scripts.benchmark.acoustic_timestep_update_microbenchmark import (
    Timer,
    make_case,
    public_variant,
    summarize,
    synchronize,
    tensor_diff,
)
from scripts.smoke.acoustic_backend_smoke import parse_dtype


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_compile_feasibility_20260531.json"
)


UpdateFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def make_update_callable(args: argparse.Namespace, mode: str) -> UpdateFn:
    free_surface_start = args.nabc if args.free_surface else 1
    nz_pml = args.nz + 2 * args.nabc
    nx_pml = args.nx + 2 * args.nabc
    z0, z1 = free_surface_start + 1, nz_pml - 2
    x0, x1 = 2, nx_pml - 2
    c1, c2 = 9.0 / 8.0, -1.0 / 24.0

    def inner(
        p: torch.Tensor,
        u: torch.Tensor,
        w: torch.Tensor,
        kappa1: torch.Tensor,
        alpha1: torch.Tensor,
    ) -> torch.Tensor:
        return (
            (1.0 - kappa1[z0:z1, x0:x1]) * p[:, z0:z1, x0:x1]
            - alpha1[z0:z1, x0:x1]
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

    if mode == "inner_only":
        return inner

    if mode == "sliced_assignment":

        def sliced_assignment(
            p: torch.Tensor,
            u: torch.Tensor,
            w: torch.Tensor,
            kappa1: torch.Tensor,
            alpha1: torch.Tensor,
        ) -> torch.Tensor:
            p_next = p.clone()
            p_next[:, z0:z1, x0:x1] = inner(p, u, w, kappa1, alpha1)
            return p_next

        return sliced_assignment

    raise ValueError(f"unsupported mode: {mode}")


def run_callable(
    args: argparse.Namespace,
    backend,
    fn: UpdateFn,
    mode: str,
    *,
    seed_offset: int,
) -> Dict[str, Any]:
    state = make_case(args, seed_offset=seed_offset)
    timer = Timer(backend)
    p_next, forward_seconds = timer.measure(
        lambda: fn(state["p"], state["u"], state["w"], state["kappa1"], state["alpha1"])
    )
    loss = p_next.pow(2).mean()
    _, backward_seconds = timer.measure(lambda: loss.backward())
    gradients = {name: state[name].grad.detach().clone() for name in ("p", "u", "w")}
    return {
        "mode": mode,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "total_seconds": forward_seconds + backward_seconds,
        "loss": loss.detach().clone(),
        "output": p_next.detach().clone(),
        "gradients": gradients,
        "finite": {
            "output": bool(torch.isfinite(p_next).all().cpu().item()),
            "p_grad": bool(torch.isfinite(gradients["p"]).all().cpu().item()),
            "u_grad": bool(torch.isfinite(gradients["u"]).all().cpu().item()),
            "w_grad": bool(torch.isfinite(gradients["w"]).all().cpu().item()),
        },
    }


def compare_runs(reference: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
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


def compile_callable(args: argparse.Namespace, fn: UpdateFn) -> UpdateFn:
    compile_kwargs: Dict[str, Any] = {}
    if args.compile_backend:
        compile_kwargs["backend"] = args.compile_backend
    if args.compile_mode:
        compile_kwargs["mode"] = args.compile_mode
    return torch.compile(fn, **compile_kwargs)


def public_compile_run(run: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **public_variant(run),
        "compile_path": run.get("compile_path"),
    }


def run_mode(args: argparse.Namespace, backend, mode: str) -> Dict[str, Any]:
    eager_fn = make_update_callable(args, mode)
    eager_reference = run_callable(args, backend, eager_fn, mode, seed_offset=0)

    if not hasattr(torch, "compile"):
        return {
            "mode": mode,
            "status": "unavailable",
            "reason": "torch.compile is not available in this PyTorch build",
            "eager_reference": public_variant(eager_reference),
        }

    try:
        synchronize(backend)
        start = time.perf_counter()
        compiled_fn = compile_callable(args, eager_fn)
        synchronize(backend)
        compile_factory_seconds = time.perf_counter() - start
    except Exception as exc:  # pragma: no cover - backend dependent
        return {
            "mode": mode,
            "status": "compile_factory_failed",
            "eager_reference": public_variant(eager_reference),
            "compile_factory_seconds": None,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=8),
        }

    try:
        synchronize(backend)
        start = time.perf_counter()
        first_eager = run_callable(args, backend, eager_fn, mode, seed_offset=1000)
        first_compiled = run_callable(args, backend, compiled_fn, mode, seed_offset=1000)
        synchronize(backend)
        first_pair_seconds = time.perf_counter() - start
    except Exception as exc:  # pragma: no cover - backend dependent
        return {
            "mode": mode,
            "status": "first_compiled_run_failed",
            "eager_reference": public_variant(eager_reference),
            "compile_factory_seconds": compile_factory_seconds,
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=12),
        }

    pairs = []
    for index in range(args.repeat):
        eager = run_callable(args, backend, eager_fn, mode, seed_offset=2000 + index)
        compiled = run_callable(args, backend, compiled_fn, mode, seed_offset=2000 + index)
        pairs.append(
            {
                "eager": public_variant(eager),
                "compiled": public_compile_run({**compiled, "compile_path": "torch.compile"}),
                "comparison": compare_runs(eager, compiled),
            }
        )

    return {
        "mode": mode,
        "status": "ok",
        "compile_factory_seconds": compile_factory_seconds,
        "first_pair_seconds": first_pair_seconds,
        "first_run": {
            "eager": public_variant(first_eager),
            "compiled": public_compile_run({**first_compiled, "compile_path": "torch.compile"}),
            "comparison": compare_runs(first_eager, first_compiled),
        },
        "summary": summarize_pairs(pairs),
        "pairs": pairs,
    }


def summarize_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "speedup": {
            "forward": summarize(pair["comparison"]["speedup"]["forward"] for pair in pairs),
            "backward": summarize(pair["comparison"]["speedup"]["backward"] for pair in pairs),
            "total": summarize(pair["comparison"]["speedup"]["total"] for pair in pairs),
        },
        "max_differences": {
            "loss_abs_diff": max(pair["comparison"]["loss_abs_diff"] for pair in pairs),
            "output_max_abs_diff": max(pair["comparison"]["output"]["max_abs_diff"] for pair in pairs),
            "p_grad_max_abs_diff": max(pair["comparison"]["gradients"]["p"]["max_abs_diff"] for pair in pairs),
            "u_grad_max_abs_diff": max(pair["comparison"]["gradients"]["u"]["max_abs_diff"] for pair in pairs),
            "w_grad_max_abs_diff": max(pair["comparison"]["gradients"]["w"]["max_abs_diff"] for pair in pairs),
        },
    }


def version_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "torch": torch.__version__,
        "has_torch_compile": hasattr(torch, "compile"),
    }
    try:
        import torch_npu  # type: ignore

        info["torch_npu"] = getattr(torch_npu, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - environment dependent
        info["torch_npu"] = None
        info["torch_npu_import_error"] = repr(exc)
    return info


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

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    results = [run_mode(args, backend, mode) for mode in modes]
    return {
        "status": "ok",
        "purpose": "Phase B torch.compile feasibility probe for acoustic AD hot path",
        "versions": version_info(),
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "repeat": args.repeat,
            "shots": args.shots,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "scale": args.scale,
            "free_surface": args.free_surface,
            "compile_backend": args.compile_backend or "torch_default",
            "compile_mode": args.compile_mode or "torch_default",
            "modes": modes,
        },
        "results": results,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--nabc", type=int, default=8)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--free-surface", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--modes", default="inner_only,sliced_assignment")
    parser.add_argument("--compile-backend", default="")
    parser.add_argument("--compile-mode", default="")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
