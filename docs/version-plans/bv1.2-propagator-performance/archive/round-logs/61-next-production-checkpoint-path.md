# Next Production Checkpoint Path

Date: 2026-06-01

## Purpose

This note records the next intended optimization path before any further code
changes. The goal is to avoid another scan-and-edit loop.

## Current Conclusion

The experimental rematerialized custom-autograd line clarified the tradeoff:

- caching all divergence terms gives speed but still uses too much memory;
- reducing saved `p/u/w` states lowers memory strongly but loses speed in the
  Python-level replay implementation;
- partial divergence-component caching is not useful enough to continue.

Therefore the next effective path should return to the production PyTorch
checkpoint kernel and look for lower-risk replay-cost reductions.

## Target File

```text
ADFWI/propagator/acoustic_kernels.py
```

The relevant production contract is:

```text
checkpoint_segments > 1
torch.utils.checkpoint.checkpoint(step_forward, ...)
```

## Allowed Optimization Types

Only consider changes that preserve the default PyTorch autograd contract:

```text
1. Hoist invariant tensor construction out of checkpoint replay when it is
   provably independent of time and does not alter gradients.
2. Reduce repeated small allocations inside `step_forward` if output, loss, and
   raw gradient parity are preserved.
3. Keep source/receiver indexing vectorized and prepared outside timestep loops
   when possible.
4. Keep `save_forward_wavefield` behavior unchanged unless the path is clearly
   opt-in and guarded by FWI-level illumination checks.
```

## Not Allowed In This Phase

Do not use this phase to:

```text
1. Promote `acoustic_custom_kernels.py` to default.
2. Promote `acoustic_kernels_bs.py` to default.
3. Change finite-difference formulas, update order, boundary condition formulas,
   source injection semantics, or receiver sampling semantics.
4. Change dtype, mixed precision, or device policy.
5. Add another broad stride/knob sweep without a measured bottleneck.
```

## Validation Gate

Every accepted production checkpoint edit must compare against the current
branch before the edit:

```text
case: reduced Marmousi2 fullshape
device: npu:0
dtype: float32
checkpoint_segments: 10
shots: 3
batch_size: 3
nx/nz/nt: 200/88/3000
loss mode: observed pressure
save_forward_wavefield: false
grad_forw_illumination: false
```

Required output:

```text
receiver p/u/w max abs diff
loss abs diff
raw vp.grad max abs diff
raw vp.grad finite check
forward time
backward time
total time
peak allocated memory
```

## Stop Condition

Stop this path if a candidate:

```text
1. changes receiver outputs or loss;
2. changes raw `vp.grad` beyond the existing rematerialized gate scale;
3. improves isolated timing but not fullshape one-iteration timing;
4. gives less than about 5% total speedup and no memory reduction;
5. requires replacing PyTorch checkpoint with custom autograd.
```

## Boundary-Saving File Position

```text
ADFWI/propagator/acoustic_kernels_bs.py
```

This file remains a research prototype for boundary-saving reconstruction. It
can be studied later, but it is not part of the immediate production checkpoint
optimization path because it uses custom global autograd state and has a
different memory/reconstruction contract.
