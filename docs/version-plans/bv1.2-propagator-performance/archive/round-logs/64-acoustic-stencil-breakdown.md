# Acoustic Stencil Breakdown

Date: 2026-06-01

## Purpose

This round moved away from small allocation candidates and measured the
finite-difference timestep pieces directly. The goal was to decide which
production checkpoint replay component is worth optimizing next.

No production code change was accepted in this round.

## New Benchmark

Added:

```text
scripts/benchmark/acoustic_timestep_stencil_breakdown.py
```

The script mirrors the tensor slices used by production `step_forward` and
times each isolated component with forward and backward:

```text
pressure_update
source_free_surface
horizontal_velocity_update
vertical_velocity_update
receiver_sampling
```

Default shape matches the reduced Marmousi2 fullshape gate:

```text
shots: 3
nx/nz: 200/88
nabc: 30
receivers: 200
device: npu:0
dtype: float32
```

## Breakdown Result

Mean total time per isolated component:

| Component | Forward mean | Backward mean | Total mean |
| --- | ---: | ---: | ---: |
| pressure update | `0.000932 s` | `0.002438 s` | `0.003371 s` |
| vertical velocity update | `0.000744 s` | `0.002068 s` | `0.002812 s` |
| horizontal velocity update | `0.000681 s` | `0.001779 s` | `0.002460 s` |
| source/free-surface | `0.000582 s` | `0.001286 s` | `0.001868 s` |
| receiver sampling | `0.000552 s` | `0.001212 s` | `0.001764 s` |

Interpretation:

```text
pressure_update > vertical_velocity_update > horizontal_velocity_update
```

Receiver sampling and source/free-surface are not the main bottleneck.

## Tested Production Candidate

Based on the breakdown, a formula-preserving candidate was tested:

```text
Hoist invariant coefficient windows out of the `step_forward` timestep loop.
```

This moved repeated slices such as:

```python
kappa1[free_surface_start + 1:nz_pml - 2, 2:nx_pml - 2]
```

to local views before the loop.

## Candidate Result

Fullshape production checkpoint gate:

| Item | Result |
| --- | ---: |
| receiver output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| total speedup | `1.004x` |
| backward speedup | `1.005x` |

Decision:

```text
Rejected.
```

The candidate is numerically correct but gives no meaningful speedup. This
suggests TorchScript/NPU already handles these coefficient views well, or the
dominant work is the actual stencil arithmetic and slice-copy operations rather
than repeated coefficient indexing.

## Next Direction

Do not continue hoisting simple coefficient windows.

The next candidate must target pressure-update stencil arithmetic or
slice-copy behavior itself. The next useful experiment is:

```text
Compare production pressure update against a mathematically identical variant
that separates divergence computation from final assignment, then test only if
the microbenchmark shows a clear pressure-update speedup.
```

If that also fails, the remaining meaningful direction is lower-level fused
stencil implementation rather than more Python/TorchScript rearrangement.
