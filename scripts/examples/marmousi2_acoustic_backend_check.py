#!/usr/bin/env python
"""Check the Marmousi2 acoustic case with the bv1.2 backend API.

This script is a lightweight bridge between the notebook Marmousi2 example and
script-style backend workflows. By default it only reads existing case files,
rebuilds the model/survey/observed-data objects, initializes the acoustic
propagator, and prints a JSON summary. It writes no notebooks, figures,
wavefields, inversion outputs, or data files.

Use ``--run-forward`` only when you intentionally want to run one selected shot
through the acoustic propagator for a stronger device check.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

import ADFWI
from ADFWI.backends import BackendUnavailableError
from ADFWI.model import AcousticModel
from ADFWI.propagator import AcousticPropagator
from ADFWI.survey import Receiver, SeismicData, Source, Survey
from ADFWI.utils import wavelet

DEFAULT_CASE_DIR = REPO_ROOT / "examples" / "acoustic" / "01-model-test" / "01-Marmousi2"


def parse_dtype(value: str) -> str:
    supported = {"float32", "float64"}
    if value not in supported:
        raise argparse.ArgumentTypeError(f"unsupported dtype {value!r}; supported: {', '.join(sorted(supported))}")
    return value


def configure_backend(args: argparse.Namespace):
    device: Optional[str] = None if args.device == "auto" else args.device
    prefer = tuple(item.strip() for item in args.prefer.split(",") if item.strip())
    return ADFWI.set_backend(device, dtype=args.dtype, fallback=args.fallback_cpu, prefer=prefer)


def cumulative_trapezoid(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.size > 1:
        out[1:] = np.cumsum((values[:-1] + values[1:]) * (0.5 * dt), dtype=np.float64).astype(np.float32)
    return out


def load_npz(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"required Marmousi2 file does not exist: {path}")
    return np.load(path, allow_pickle=True)


def optional_bound(value: np.ndarray) -> Optional[Tuple[float, float]]:
    arr = np.asarray(value)
    if arr.dtype == object:
        values = arr.tolist()
        if values is None or any(item is None for item in values):
            return None
    if arr.size != 2:
        return None
    return float(arr[0]), float(arr[1])


def build_survey(obs_npz: Any, f0: float) -> Survey:
    nt = int(obs_npz["nt"])
    dt = float(obs_npz["dt"])
    src_loc = np.asarray(obs_npz["src_loc"], dtype=np.int64)
    rcv_loc = np.asarray(obs_npz["rcv_loc"], dtype=np.int64)
    src_type = np.asarray(obs_npz["src_type"]).astype(str)
    rcv_type = np.asarray(obs_npz["rcv_type"]).astype(str)

    _, src_v = wavelet(nt, dt, f0, amp0=1)
    src_v = cumulative_trapezoid(src_v.astype(np.float32), dt)
    source = Source(nt=nt, dt=dt, f0=f0)
    for (src_x, src_z), src_kind in zip(src_loc, src_type):
        source.add_source(int(src_x), int(src_z), src_v, src_type=str(src_kind))

    receiver = Receiver(nt=nt, dt=dt)
    for (rcv_x, rcv_z), rcv_kind in zip(rcv_loc, rcv_type):
        receiver.add_receiver(int(rcv_x), int(rcv_z), rcv_type=str(rcv_kind))
    return Survey(source, receiver)


def build_model(model_npz: Any, args: argparse.Namespace) -> AcousticModel:
    vp = np.asarray(model_npz["vp"], dtype=np.float32)
    rho = np.asarray(model_npz["rho"], dtype=np.float32)
    return AcousticModel(
        float(model_npz["ox"]),
        float(model_npz["oz"]),
        int(model_npz["nx"]),
        int(model_npz["nz"]),
        float(model_npz["dx"]),
        float(model_npz["dz"]),
        vp,
        rho,
        vp_bound=optional_bound(model_npz["vp_bound"]),
        rho_bound=optional_bound(model_npz["rho_bound"]),
        vp_grad=False,
        rho_grad=False,
        auto_update_rho=False,
        free_surface=bool(model_npz["free_surface"]),
        abc_type=args.abc_type,
        abc_jerjan_alpha=args.abc_jerjan_alpha,
        nabc=int(model_npz["nabc"]),
    )


def array_summary(array: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bool(np.isfinite(array).all()),
        "min": float(np.nanmin(array)),
        "max": float(np.nanmax(array)),
        "norm": float(np.linalg.norm(array.reshape(-1))),
    }


def observed_summary(obs_data: SeismicData) -> Dict[str, Any]:
    data = obs_data.data or {}
    summary: Dict[str, Any] = {}
    for key in ("p", "u", "w"):
        if key in data:
            summary[key] = array_summary(np.asarray(data[key]))
    return summary


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "device": str(detached.device),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(detached).all().item()),
        "min": float(detached.min().cpu().item()),
        "max": float(detached.max().cpu().item()),
    }


def maybe_run_forward(propagator: AcousticPropagator, backend, args: argparse.Namespace) -> Dict[str, Any] | None:
    if not args.run_forward:
        return None
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        record = propagator.forward(
            shot_index=np.array([args.shot_index], dtype=np.int64),
            checkpoint_segments=args.checkpoint_segments,
        )
    if backend.name in {"cuda", "npu"}:
        backend.synchronize()
    seconds = time.perf_counter() - start
    pressure = record["p"].detach()
    if not torch.isfinite(pressure).all():
        raise RuntimeError("single-shot forward pressure contains NaN or Inf")
    return {
        "shot_index": args.shot_index,
        "seconds": seconds,
        "pressure": tensor_summary(pressure),
    }


def run_check(args: argparse.Namespace) -> Dict[str, Any]:
    backend = configure_backend(args)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    case_dir = args.case_dir.resolve()
    model_path = case_dir / "data" / "model" / args.model_file
    obs_path = case_dir / "data" / "waveform" / "obs_data.npz"
    model_npz = load_npz(model_path)
    obs_npz = load_npz(obs_path)

    survey = build_survey(obs_npz, f0=args.f0)
    model = build_model(model_npz, args)
    obs_data = SeismicData(survey)
    obs_data.load(str(obs_path))
    propagator = AcousticPropagator(model, survey)
    forward_summary = maybe_run_forward(propagator, backend, args)

    observed = observed_summary(obs_data)
    pressure = observed.get("p")
    if pressure is None:
        raise RuntimeError("observed Marmousi2 data does not contain pressure key 'p'")
    if not pressure["finite"] or not math.isfinite(pressure["norm"]) or pressure["norm"] <= 0.0:
        raise RuntimeError("observed pressure is invalid or zero")

    return {
        "status": "ok",
        "case": "Marmousi2 acoustic",
        "case_dir": str(case_dir),
        "model_file": args.model_file,
        "backend": ADFWI.backend_diagnostics(),
        "model": {
            "vp": tensor_summary(model.vp),
            "rho": tensor_summary(model.rho),
            "nx": model.nx,
            "nz": model.nz,
            "dx": model.dx,
            "dz": model.dz,
            "nabc": model.nabc,
            "free_surface": model.free_surface,
        },
        "survey": {
            "shots": survey.source.num,
            "receivers": survey.receiver.num,
            "nt": survey.source.nt,
            "dt": survey.source.dt,
            "source_x_range": [int(np.min(survey.source.get_loc()[:, 0])), int(np.max(survey.source.get_loc()[:, 0]))],
            "receiver_x_range": [int(np.min(survey.receiver.get_loc()[:, 0])), int(np.max(survey.receiver.get_loc()[:, 0]))],
        },
        "observed": observed,
        "propagator": {
            "device": str(propagator.device),
            "dtype": str(propagator.dtype).replace("torch.", ""),
            "damp": tensor_summary(propagator.damp),
        },
        "single_shot_forward": forward_summary,
        "seed": args.seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, default=DEFAULT_CASE_DIR)
    parser.add_argument("--model-file", default="init_model.npz", choices=("init_model.npz", "true_model.npz"))
    parser.add_argument("--device", default="auto", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu", help="auto-selection priority used when --device auto")
    parser.add_argument("--fallback-cpu", action="store_true", help="fallback explicit unavailable accelerator requests to CPU")
    parser.add_argument("--dtype", type=parse_dtype, default="float32")
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--f0", type=float, default=5.0)
    parser.add_argument("--abc-type", default="PML")
    parser.add_argument("--abc-jerjan-alpha", type=float, default=0.007)
    parser.add_argument("--run-forward", action="store_true", help="run one selected shot after the read-only case check")
    parser.add_argument("--shot-index", type=int, default=0)
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_check(args)
    except BackendUnavailableError as exc:
        print(json.dumps({"status": "unavailable", "error": str(exc)}, indent=2), file=sys.stderr)
        return 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": repr(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
