# Acoustic Checkpoint Segment Sweep

Date: 2026-05-31

## Purpose

This round evaluates the remaining low-risk Phase B option:

```text
checkpoint / rematerialization time-memory tradeoff
```

No propagator code was changed. The benchmark sweeps production
`checkpoint_segments` values and compares each segmented run against
`checkpoint_segments=1`.

Benchmark script:

```text
scripts/benchmark/acoustic_checkpoint_segment_sweep.py
```

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_segment_sweep.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 2 \
  --segments 1,2,4,8 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_checkpoint_segment_sweep_20260531.json
```

## Case

| Field | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Model | `nx=100`, `nz=50` |
| Boundary | `nabc=20` |
| Time samples | `nt=800` |
| Shots | `1` |
| Receivers | `3` |
| Loss component | `p` |
| Forward wavefield | saved |

## Numerical Result

Compared with `checkpoint_segments=1`:

| Segments | loss abs diff | pressure max abs diff | `vp.grad` max abs diff | `vp.grad` max rel diff |
| --- | ---: | ---: | ---: | ---: |
| `2` | `0.0` | `0.0` | `1.3877787807814457e-17` | `3.737431484296394e-07` |
| `4` | `0.0` | `0.0` | `6.938893903907228e-18` | `5.421010769168788e-07` |
| `8` | `0.0` | `0.0` | `6.938893903907228e-18` | `8.204478376683255e-07` |

Receiver outputs and scalar loss are identical. Gradient differences are at
floating-point recomputation level.

## Timing Result

Mean across two measured repeats:

| Segments | Forward mean | Backward mean | Total mean |
| --- | ---: | ---: | ---: |
| `1` | `1.7626216216012836 s` | `4.920417697168887 s` | `6.68303931877017 s` |
| `2` | `1.2582435784861445 s` | `7.269093309529126 s` | `8.52733688801527 s` |
| `4` | `1.2500287806615233 s` | `7.355418362654746 s` | `8.605447143316269 s` |
| `8` | `1.241103233769536 s` | `7.405091587454081 s` | `8.646194821223617 s` |

Relative to `segments=1`, segmented checkpointing:

- reduces measured forward time;
- increases backward time substantially because rematerialization recomputes
  forward segments during backward;
- increases total time for this NPU case.

## Memory Note

The benchmark records `backend.memory_allocated()`, but this is not peak memory.
It is not sufficient to prove memory savings. A peak-memory-specific benchmark
would be needed before making memory claims.

## Decision

Reject checkpoint segmentation as a speed optimization for the current NPU
performance route:

```text
Keep checkpoint_segments=1 for performance-focused acoustic runs when memory
is sufficient.
```

Segmented checkpointing remains useful as a memory-control option for larger
cases that cannot fit with `checkpoint_segments=1`, but it should not be used
as a default speed path.

## Next Direction

The low-risk PyTorch-native Phase B options have now been checked:

- opt-in forward-wavefield skip: valid for `forw_illumination=False`, not a
  default path;
- functional reconstruction update style: rejected, slower;
- checkpoint segmentation: rejected as speed path, useful only for memory.

Next work should not proceed as another small code tweak. The remaining routes
are higher risk and need an explicit choice:

1. custom autograd / adjoint-state prototype for a tiny acoustic operator;
2. backend compile/fusion feasibility probe;
3. stop acoustic default-path optimization and move to a different module or
   elastic profiling.
