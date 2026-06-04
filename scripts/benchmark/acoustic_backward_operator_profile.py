#!/usr/bin/env python
"""Profile acoustic backward operators on NPU/CPU.

This is a Phase B diagnostic. It profiles the default acoustic propagator path
without changing kernels, then records top backward operators and wall-clock
forward/backward timings. Use ``--pressure-only`` to profile the current
AcousticFWI pressure-loss production route. The default shape is intentionally
smaller than the reduced Marmousi2 validation case so profiler traces stay
manageable.
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

from ADFWI.backends import configure_backend
from ADFWI.propagator import AcousticPropagator
from ADFWI.survey import Receiver, Source, Survey
from scripts.smoke.acoustic_backend_smoke import build_model, parse_dtype, ricker_wavelet


DEFAULT_OUTPUT = (
    REPO_ROOT
    / "docs"
    / "version-plans"
    / "bv1.2-propagator-performace-deepwave"
    / "develope"
    / "acoustic_backward_operator_profile_20260531.json"
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


def tensor_summary(value: torch.Tensor) -> Dict[str, Any]:
    detached = value.detach()
    return {
        "shape": list(detached.shape),
        "device": str(detached.device),
        "dtype": str(detached.dtype).replace("torch.", ""),
        "finite": bool(torch.isfinite(detached).all().cpu().item()),
        "norm": float(torch.linalg.norm(detached.reshape(-1)).cpu().item()),
    }


def build_profile_survey(args: argparse.Namespace) -> Survey:
    source = Source(nt=args.nt, dt=args.dt, f0=args.f0)
    src_x = np.array([i for i in range(2, args.nx - 1, args.source_spacing)], dtype=np.int64)[: args.shots]
    src_z = np.full_like(src_x, args.source_depth)
    src_wavelet = ricker_wavelet(args.nt, args.dt, args.f0)
    for x, z in zip(src_x, src_z):
        source.add_source(int(x), int(z), src_wavelet, src_type="mt")

    receiver = Receiver(nt=args.nt, dt=args.dt)
    if args.receivers >= args.nx:
        rcv_x = np.arange(args.nx, dtype=np.int64)
    else:
        rcv_x = np.linspace(0, args.nx - 1, args.receivers, dtype=np.int64)
    rcv_z = np.full_like(rcv_x, args.receiver_depth)
    receiver.add_receivers(rcv_x, rcv_z, rcv_type="pr")
    return Survey(source, receiver)


def build_case(args: argparse.Namespace):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(args.nx, args.nz, args.dx, args.dz, args.nabc)
    survey = build_profile_survey(args)
    propagator = AcousticPropagator(model, survey)
    model.zero_grad(set_to_none=True)
    return model, propagator


def event_payload(event, *, index: int) -> Dict[str, Any]:
    payload = {
        "rank": index,
        "key": event.key,
        "count": int(event.count),
        "cpu_time_total_us": float(event.cpu_time_total),
        "cpu_time_self_us": float(event.self_cpu_time_total),
    }
    for attr in (
        "device_time_total",
        "self_device_time_total",
        "device_memory_usage",
        "self_device_memory_usage",
        "npu_time_total",
        "self_npu_time_total",
        "cuda_time_total",
        "self_cuda_time_total",
    ):
        if hasattr(event, attr):
            try:
                payload[f"{attr}_us"] = float(getattr(event, attr))
            except Exception:
                pass
    return payload


def run_once(args: argparse.Namespace, backend) -> Dict[str, Any]:
    timer = Timer(backend)
    model, propagator = build_case(args)
    shot_index = np.arange(args.shots, dtype=np.int64)

    record, forward_seconds = timer.measure(
        lambda: propagator.forward(
            shot_index=shot_index,
            checkpoint_segments=args.checkpoint_segments,
            save_forward_wavefield=args.save_forward_wavefield,
            pressure_only=args.pressure_only,
        )
    )
    loss = record[args.loss_component].pow(2).mean()

    synchronize(backend)
    use_device = "npu" if backend.name == "npu" else None
    with torch.autograd.profiler.profile(
        use_device=use_device,
        use_cpu=True,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        loss.backward()
        synchronize(backend)
        backward_seconds = time.perf_counter() - start

    if model.vp.grad is None:
        raise RuntimeError("model.vp.grad is None after backward")
    if not torch.isfinite(model.vp.grad).all():
        raise RuntimeError("model.vp.grad contains NaN or Inf")

    events = prof.key_averages()
    sort_key = "self_device_time_total" if backend.name == "npu" else "self_cpu_time_total"
    top_events = sorted(events, key=lambda event: getattr(event, sort_key, 0.0), reverse=True)[: args.topk]
    return {
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "loss": float(loss.detach().cpu().item()),
        "record": {key: tensor_summary(record[key]) for key in ("p", "u", "w")},
        "vp_grad": tensor_summary(model.vp.grad),
        "top_events_sort": sort_key,
        "top_events": [event_payload(event, index=index + 1) for index, event in enumerate(top_events)],
        "memory_allocated": backend.memory_allocated(),
    }


def summarize(values: Iterable[float]) -> Dict[str, float]:
    data = list(values)
    return {
        "min": float(min(data)),
        "max": float(max(data)),
        "mean": float(sum(data) / len(data)),
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

    if args.warmup:
        model, propagator = build_case(args)
        timer = Timer(backend)
        record, _ = timer.measure(
            lambda: propagator.forward(
                shot_index=np.arange(args.shots, dtype=np.int64),
                checkpoint_segments=args.checkpoint_segments,
                save_forward_wavefield=args.save_forward_wavefield,
                pressure_only=args.pressure_only,
            )
        )
        loss = record[args.loss_component].pow(2).mean()
        _, _ = timer.measure(lambda: loss.backward())
        del model, propagator, record, loss

    repeats = [run_once(args, backend) for _ in range(args.repeat)]
    return {
        "status": "ok",
        "purpose": "Phase B acoustic backward operator profile",
        "backend": backend.diagnostics(),
        "config": {
            "device": args.device,
            "prefer": args.prefer,
            "fallback_cpu": args.fallback_cpu,
            "dtype": str(args.dtype).replace("torch.", ""),
            "seed": args.seed,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "checkpoint_segments": args.checkpoint_segments,
            "save_forward_wavefield": args.save_forward_wavefield,
            "pressure_only": args.pressure_only,
            "loss_component": args.loss_component,
            "shots": args.shots,
            "receivers": args.receivers,
            "source_spacing": args.source_spacing,
            "source_depth": args.source_depth,
            "receiver_depth": args.receiver_depth,
            "nx": args.nx,
            "nz": args.nz,
            "nabc": args.nabc,
            "nt": args.nt,
            "dx": args.dx,
            "dz": args.dz,
            "dt": args.dt,
            "f0": args.f0,
            "topk": args.topk,
            "record_shapes": args.record_shapes,
            "profile_memory": args.profile_memory,
        },
        "summary": {
            "forward_seconds": summarize(run["forward_seconds"] for run in repeats),
            "backward_seconds": summarize(run["backward_seconds"] for run in repeats),
            "loss": summarize(run["loss"] for run in repeats),
        },
        "repeats": repeats,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="npu:0", help="cpu, npu:0, cuda:0, or auto")
    parser.add_argument("--prefer", default="npu,cpu")
    parser.add_argument("--fallback-cpu", action="store_true")
    parser.add_argument("--dtype", default=torch.float32, type=parse_dtype)
    parser.add_argument("--seed", type=int, default=20240523)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--checkpoint-segments", type=int, default=1)
    parser.add_argument("--save-forward-wavefield", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pressure-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loss-component", choices=("p", "u", "w"), default="p")
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--receivers", type=int, default=3)
    parser.add_argument("--source-spacing", type=int, default=5)
    parser.add_argument("--source-depth", type=int, default=2)
    parser.add_argument("--receiver-depth", type=int, default=2)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--nz", type=int, default=50)
    parser.add_argument("--nabc", type=int, default=20)
    parser.add_argument("--nt", type=int, default=800)
    parser.add_argument("--dx", type=float, default=40.0)
    parser.add_argument("--dz", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.003)
    parser.add_argument("--f0", type=float, default=5.0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.shots <= 0:
        parser.error("--shots must be positive")
    if args.receivers <= 0:
        parser.error("--receivers must be positive")
    if args.source_spacing <= 0:
        parser.error("--source-spacing must be positive")

    report = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
