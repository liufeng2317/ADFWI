# Acoustic Default Backward Operator Profile

Date: 2026-05-31

## Purpose

This round returns from the opt-in wavefield policy branch to the main Phase B
line:

```text
Profile the default acoustic backward path at finer granularity.
```

No kernel behavior was changed. A profiling harness was added:

```text
scripts/benchmark/acoustic_backward_operator_profile.py
```

The script uses `torch.autograd.profiler.profile(use_device="npu")` to collect
operator-level backward events and synchronized wall-clock timings.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_backward_operator_profile.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --topk 20 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_backward_operator_profile_20260531.json
```

## Case

This is a representative operator-profile case, not the full reduced
Marmousi2 workflow.

| Field | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Model | `nx=100`, `nz=50` |
| Boundary | `nabc=20` |
| Time samples | `nt=800` |
| Shots | `1` |
| Receivers | `3` |
| `checkpoint_segments` | `1` |
| `save_forward_wavefield` | `True` |
| loss component | `p` |

Profiler overhead is high, so wall time in this record should not be compared
directly with normal benchmark wall time. The useful result is the relative
operator distribution.

## Numerical Sanity

| Metric | Value |
| --- | --- |
| loss | `3.101359880020027e-07` |
| pressure finite | `True` |
| `vp.grad` finite | `True` |
| `vp.grad` norm | `8.007337570781203e-10` |

## Top Backward Operators

Sorted by `self_device_time_total_us`.

| Rank | Operator | Count | Self device time |
| --- | --- | ---: | ---: |
| 1 | `aten::copy_` | `58351` | `3.538005 s` |
| 2 | `aclnnInplaceCopy` | `58349` | `3.038150 s` |
| 3 | `aclnnInplaceZero` | `53543` | `2.890947 s` |
| 4 | `aten::slice_backward` | `47941` | `2.863478 s` |
| 5 | `autograd::engine::evaluate_function: SliceBackward0` | `47941` | `2.449706 s` |
| 6 | `aten::zero_` | `53543` | `2.076264 s` |
| 7 | `aten::zeros` | `48743` | `2.062465 s` |
| 8 | `empty_tensor` | `86343` | `2.031409 s` |
| 9 | `aten::slice` | `47941` | `1.345226 s` |
| 10 | `SliceBackward0` | `47941` | `1.333415 s` |
| 11 | `aclnnInplaceAdd` | `17580` | `0.920714 s` |
| 12 | `aten::as_strided` | `53550` | `0.803520 s` |

Approximate grouping over the top-20 self device time:

| Group | Time | Share |
| --- | ---: | ---: |
| slice/view backward | `9.162013 s` | `31.18%` |
| copy/write/index put | `7.358473 s` | `25.05%` |
| zero/allocation | `6.170138 s` | `21.00%` |
| simple math | `2.366001 s` | `8.05%` |

## Interpretation

Default acoustic backward is dominated by autograd overhead from the timestep
implementation pattern:

```text
many sliced assignments -> CopySlices / SliceBackward0 -> copy / zero / allocation
```

The profile does not point to a single expensive physical math operation. It
points to the cost of recording and replaying thousands of sliced tensor
updates in PyTorch autograd.

This is consistent with the previous Phase A result where backward dominated
the reduced FWI iteration.

## Optimization Implication

The next default-path optimization should not focus on loss evaluation,
gradient processing, wrapper cleanup, or plotting. The next candidate must
target the timestep update structure that creates many slice backward and copy
events.

Candidate directions, in increasing risk:

1. Build a tiny controlled experiment for one timestep update style and compare
   sliced assignment vs functional reconstruction for output/gradient parity
   and operator profile.
2. Explore whether local temporary tensors can reduce repeated zero/copy
   backward work without changing equations.
3. Treat custom autograd or adjoint-state as research-only until lower-risk
   update-structure experiments are measured.

## Decision

Continue the main Phase B line. The next task should be a bounded acoustic
timestep-update microbenchmark, not a direct kernel rewrite.

Required next validation:

- compare output tensors;
- compare raw `vp.grad`;
- compare operator profile and wall time;
- do not merge any timestep rewrite without reduced FWI numerical parity.
