#!/usr/bin/env python
"""Public backend API smoke test for bv1.2.

This script validates the lightweight researcher-facing backend entry points
without running wave propagation or writing output files. It checks that the
public top-level API configures a backend, reports diagnostics, creates tensors
on the expected dtype/device, and restores the previous backend after a scoped
CPU override.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import ADFWI
from ADFWI.backends import BackendUnavailableError, use_backend


def parse_device(value: str) -> Optional[str]:
    """Return None for auto-selection, otherwise the requested device string."""
    text = value.strip()
    if text.lower() in {"auto", "none"}:
        return None
    return text


def parse_prefer(value: str) -> Tuple[str, ...]:
    """Parse comma-separated backend priority such as npu,cpu."""
    prefer = tuple(item.strip() for item in value.split(",") if item.strip())
    if not prefer:
        raise argparse.ArgumentTypeError("prefer must contain at least one backend family")
    return prefer


def expected_device_type(diagnostics: Dict[str, Any]) -> str:
    """Return the torch device type expected from backend diagnostics."""
    return str(diagnostics["device"]).split(":", 1)[0]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    requested_device = parse_device(args.device)
    prefer = parse_prefer(args.prefer)

    backend = ADFWI.set_backend(
        requested_device,
        dtype=args.dtype,
        fallback=args.fallback_cpu,
        prefer=prefer,
    )
    diagnostics = ADFWI.backend_diagnostics()

    tensor = ADFWI.backend().ones((2, 3))
    if tensor.dtype != backend.dtype:
        raise RuntimeError(f"tensor dtype {tensor.dtype} does not match backend dtype {backend.dtype}")
    if tensor.device.type != expected_device_type(diagnostics):
        raise RuntimeError(f"tensor device {tensor.device} does not match diagnostics {diagnostics['device']}")

    before_scoped = ADFWI.backend_diagnostics()
    with use_backend("cpu", dtype=args.dtype) as scoped:
        scoped_tensor = ADFWI.backend().zeros((1,))
        if scoped.name != "cpu":
            raise RuntimeError(f"scoped backend should be cpu, got {scoped.name}")
        if scoped_tensor.device.type != "cpu":
            raise RuntimeError(f"scoped tensor should be on CPU, got {scoped_tensor.device}")
    after_scoped = ADFWI.backend_diagnostics()
    if after_scoped["device"] != before_scoped["device"] or after_scoped["dtype"] != before_scoped["dtype"]:
        raise RuntimeError("scoped backend did not restore the previous active backend")

    return {
        "status": "ok",
        "request": {
            "device": args.device,
            "resolved_device_arg": requested_device,
            "prefer": list(prefer),
            "dtype": args.dtype,
            "fallback_cpu": args.fallback_cpu,
        },
        "backend": diagnostics,
        "tensor_check": {
            "shape": list(tensor.shape),
            "device": str(tensor.device),
            "dtype": str(tensor.dtype).replace("torch.", ""),
            "sum": float(tensor.detach().cpu().sum().item()),
        },
        "scoped_override": {
            "inside_device": "cpu",
            "restored_device": after_scoped["device"],
            "restored_dtype": after_scoped["dtype"],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="cpu, cuda:0, npu:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority, e.g. npu,cpu or cuda,cpu")
    parser.add_argument("--dtype", default="float32", help="backend dtype string accepted by ADFWI.set_backend")
    parser.add_argument("--fallback-cpu", action="store_true", help="fall back to CPU when an explicit accelerator is unavailable")
    args = parser.parse_args()

    try:
        result = run(args)
    except BackendUnavailableError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, indent=2, sort_keys=True))
        return 2
    except Exception as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, indent=2, sort_keys=True))
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
