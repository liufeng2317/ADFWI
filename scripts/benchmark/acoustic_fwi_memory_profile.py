#!/usr/bin/env python
"""Measure acoustic FWI iteration memory for production/custom propagator paths."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.benchmark import acoustic_fwi_iteration_profile as profile


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performance"
    / "acoustic_fwi_memory_profile_20260531.json"
)


def synchronize(backend) -> None:
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()


def memory_api(torch_module, backend):
    if backend.name == "npu":
        return torch_module.npu
    if backend.name == "cuda":
        return torch_module.cuda
    return None


def memory_allocated(torch_module, backend) -> Optional[int]:
    api = memory_api(torch_module, backend)
    if api is None or not hasattr(api, "memory_allocated"):
        return None
    return int(api.memory_allocated())


def max_memory_allocated(torch_module, backend) -> Optional[int]:
    api = memory_api(torch_module, backend)
    if api is None or not hasattr(api, "max_memory_allocated"):
        return None
    return int(api.max_memory_allocated())


def reset_peak_memory(torch_module, backend) -> None:
    api = memory_api(torch_module, backend)
    if api is None:
        return
    if hasattr(api, "empty_cache"):
        api.empty_cache()
    if hasattr(api, "reset_peak_memory_stats"):
        api.reset_peak_memory_stats()
    elif hasattr(api, "reset_max_memory_allocated"):
        api.reset_max_memory_allocated()


def bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return value / (1024.0 * 1024.0)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    setup_start = time.perf_counter()
    rt = profile.forward_modeling.import_runtime_modules()
    backend = rt["ADFWI"].set_backend(args.device, dtype=args.dtype, fallback=args.fallback_cpu)
    observed_report = profile.ensure_observed_data(rt, args, backend)
    fwi, vp_init = profile.build_fwi_state(rt, args, backend)
    synchronize(backend)
    setup_seconds = time.perf_counter() - setup_start

    torch = rt["torch"]
    setup_memory = memory_allocated(torch, backend)
    reset_peak_memory(torch, backend)
    synchronize(backend)
    before_iteration = memory_allocated(torch, backend)
    timer = profile.Timer(backend)
    iteration = profile.run_one_iteration(fwi, args, timer)
    synchronize(backend)
    after_iteration = memory_allocated(torch, backend)
    peak_iteration = max_memory_allocated(torch, backend)
    vp_update_norm = float(
        torch.linalg.norm(
            (fwi.model.vp.detach() - torch.as_tensor(vp_init, device=fwi.model.vp.device, dtype=fwi.model.vp.dtype)).reshape(-1)
        )
        .cpu()
        .item()
    )

    return {
        "status": "ok",
        "case": profile.VALIDATION_CASES[args.validation_case]["case"],
        "purpose": "Acoustic FWI one-iteration timing and peak memory profile",
        "backend": rt["ADFWI"].backend_diagnostics(),
        "shape": {
            "shots": args.shots,
            "receivers": fwi.propagator.rcv_n,
            "nt": fwi.propagator.nt,
            "nx": fwi.model.nx,
            "nz": fwi.model.nz,
            "checkpoint_segments": args.checkpoint_segments,
            "save_forward_wavefield": args.save_forward_wavefield,
            "grad_forw_illumination": args.grad_forw_illumination,
            "use_custom_chunk_backward": args.use_custom_chunk_backward,
            "gradient_processor": args.gradient_processor,
        },
        "setup_seconds": setup_seconds,
        "observed_data": observed_report,
        "memory": {
            "setup_allocated_bytes": setup_memory,
            "before_iteration_allocated_bytes": before_iteration,
            "after_iteration_allocated_bytes": after_iteration,
            "peak_iteration_allocated_bytes": peak_iteration,
            "setup_allocated_mib": bytes_to_mib(setup_memory),
            "before_iteration_allocated_mib": bytes_to_mib(before_iteration),
            "after_iteration_allocated_mib": bytes_to_mib(after_iteration),
            "peak_iteration_allocated_mib": bytes_to_mib(peak_iteration),
        },
        "iteration": iteration,
        "vp_update_norm": vp_update_norm,
    }


def build_parser(argv: Optional[list[str]] = None) -> argparse.ArgumentParser:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--validation-case", choices=tuple(profile.VALIDATION_CASES), default="reduced")
    pre_args, _ = pre_parser.parse_known_args(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    profile.add_arguments(parser, pre_args.validation_case)
    parser.set_defaults(result_json=DEFAULT_OUTPUT)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser(argv)
    args = parser.parse_args(argv)
    profile.forward_modeling.validate_case_args(parser, args)
    if args.iterations != 1:
        parser.error("memory profile expects --iterations 1")
    report = run(args)
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=profile.forward_modeling.json_default) + "\n"
    )
    print(
        json.dumps(
            {
                "memory": report["memory"],
                "timing_total": report["iteration"]["timing_total"],
                "timings": report["iteration"]["timings"],
                "loss": report["iteration"]["loss"],
                "raw_grad_finite": report["iteration"]["raw_grad_finite"],
                "grad_finite_after_processing": report["iteration"]["grad_finite_after_processing"],
                "vp_finite_after_optimizer": report["iteration"]["vp_finite_after_optimizer"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.result_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
