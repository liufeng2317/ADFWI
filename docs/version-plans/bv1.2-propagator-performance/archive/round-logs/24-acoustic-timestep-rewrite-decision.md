# 24 - Acoustic Timestep Rewrite Decision

Date: 2026-05-31

## Purpose

Close the repeated acoustic backward-analysis loop and decide whether the
current branch should attempt a bounded acoustic timestep state-update rewrite.

This is a decision record, not a new benchmark and not a kernel change.

## Evidence Already Collected

The same bottleneck has now been confirmed through multiple independent
records:

| Record | Scope | Key conclusion |
| --- | --- | --- |
| `12-acoustic-default-backward-operator-profile.md` | small operator profile | backward dominated by `slice_backward`, `SliceBackward0`, `copy_`, `zero_`, `zeros`, `empty_tensor` |
| `13-acoustic-timestep-update-microbenchmark.md` | isolated pressure update | `torch.cat` reconstruction preserved gradients but was slower |
| `14-acoustic-checkpoint-segment-sweep.md` | checkpoint policy | `checkpoint_segments=1` remains fastest when memory is sufficient |
| `15-acoustic-compile-feasibility.md` | compile route | default NPU compile path blocked by missing `triton`; eager backend gives no useful speedup |
| `17-acoustic-pressure-inner-state-probe.md` | recurrent state rewrite probe | Python-level inner-state recurrence hit autograd in-place version constraints |
| `22-acoustic-full-record-iteration-profile.md` | full-record FWI iteration | backward `16.93 s` and forward `9.83 s` dominate a `28.27 s` measured iteration |
| `23-acoustic-full-record-backward-operator-profile.md` | full-record operator profile | full-record shape confirms the same slice/copy/zero/allocation autograd overhead |

This is enough evidence to stop profiling the same issue.

## Current Kernel Structure

The hot path in `ADFWI/propagator/acoustic_kernels.py` updates recurrent state
inside the timestep loop:

```text
p[:, interior] = ...
u[:, interior] = ...
w[:, interior] = ...
receiver[:, it, :] = ...
```

This structure is clear and numerically established, but PyTorch autograd
records many sliced assignment, copy, zero, and allocation operations during
backward.

The profiler results show that the cost is not a single expensive finite
difference math operation. The cost is the autograd representation of the
recurrent sliced-update pattern.

## Low-Risk Options Status

| Candidate | Status | Reason |
| --- | --- | --- |
| `checkpoint_segments == 1` no-checkpoint path | accepted | exact numerical parity and faster when no segmentation is requested |
| source index hoist | accepted | exact numerical parity and small speedup |
| `save_forward_wavefield=False` when illumination is disabled | accepted as opt-in | exact parity for valid policy, but not a default behavior |
| `torch.cat` functional reconstruction | rejected | exact but slower |
| receiver list/stack recording | rejected | isolated faster but full FWI slower |
| `torch.compile` | rejected for current NPU environment | optimizing backend unavailable |
| checkpoint segmentation for speed | rejected | useful for memory control, slower for measured NPU case |
| Python-level inner-state recurrence | rejected for default path | hit autograd in-place version constraints |
| `TorchGradProcessor` as default | rejected | reduced case faster, full-record 10-iter slower |

## Decision

Do not attempt another small acoustic timestep state-update rewrite in this
branch.

Reason:

```text
The remaining changes that could materially reduce backward overhead require
changing the autograd representation of the recurrent wave equation update.
That is no longer a local performance cleanup; it is a numerical algorithm /
custom-gradient problem.
```

The current branch should preserve the accepted safe changes and stop the
acoustic micro-optimization loop here.

## What Would Be Required For A Future Rewrite

A future acoustic timestep rewrite should be isolated in a separate research
task or branch and treated as a custom-gradient/adjoint validation problem.

Minimum acceptance gates:

1. Tiny CPU case comparing forward `p/u/w` exactly or within explicit tolerance.
2. Tiny CPU case comparing raw `vp.grad` against current autograd.
3. NPU tiny case comparing forward `p/u/w` and raw `vp.grad`.
4. Reduced Marmousi2 FWI 1-iteration comparison:
   - loss;
   - raw gradient norm;
   - processed gradient norm;
   - `vp_update_norm`.
5. Reduced Marmousi2 10-iteration comparison.
6. Full-record short validation before baseline promotion.

Without these gates, a timestep rewrite risks silently changing the inversion
trajectory.

## Next Direction

For this branch:

```text
Stop acoustic kernel micro-optimization and write a branch-level acoustic
performance closeout.
```

Recommended follow-up after closeout:

```text
Open a separate research branch only if we want to prototype a custom
autograd/adjoint acoustic operator. Otherwise, move performance work to a
different measured bottleneck.
```
