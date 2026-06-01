# Production Checkpoint Candidate Round

Date: 2026-06-01

## Purpose

This round tried the highest-probability low-risk production checkpoint
candidates after the custom-autograd memory work stalled. The goal was to find
an optimization that preserves the production PyTorch autograd contract.

Target:

```text
ADFWI/propagator/acoustic_kernels.py
checkpoint_segments > 1
torch.utils.checkpoint.checkpoint(step_forward, ...)
```

## Gate

Fullshape reduced Marmousi2 production checkpoint gate:

```text
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

## Candidate A: Non-Reentrant Checkpoint

Temporary change:

```python
checkpoint(..., use_reentrant=False)
```

Result:

```text
Rejected.
```

The gate failed during backward on NPU/TorchScript replay at receiver `w`
sampling:

```text
RuntimeError in step_forward:
rcv_w[:, it, :] = w[:, rcv_z, rcv_x]
```

This means non-reentrant checkpoint is not a safe production candidate for the
current scripted acoustic kernel on the tested NPU stack.

## Candidate B: Pre-Scale Source Wavelet

Temporary change:

```text
forward_kernel: src_v = dt * src_v before chunking
step_forward: remove per-time-step dt multiplication during source injection
```

Numerical result:

| Item | Difference |
| --- | ---: |
| receiver output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| raw `vp.grad` max rel diff | `0.0` |

Timing result:

```text
total speedup: 1.025x
backward speedup: 1.022x
forward speedup: 1.038x
```

Decision:

```text
Rejected. Correct but below the 5% total-speedup acceptance threshold and no
memory reduction.
```

## Candidate C: Empty Receiver Allocation

Temporary change:

```text
Use torch.empty instead of torch.zeros for receiver tensors that are fully
written by receiver sampling.
```

Numerical result:

| Item | Difference |
| --- | ---: |
| receiver output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| raw `vp.grad` max rel diff | `0.0` |

Timing result:

```text
total speedup: 1.011x
backward speedup: 1.013x
forward speedup: 0.986x
```

Decision:

```text
Rejected. Correct but too small and does not reduce peak memory.
```

## Final Decision

No production code change was accepted in this round.

The temporary code edits were reverted. Benchmark JSON records are kept for
the accepted gate runs so these low-yield candidates are not repeated.

## Interpretation

The checkpoint replay cost is not dominated by these small per-step or
allocation overheads:

```text
forward-wavefield placeholder allocation
source dt scaling
receiver zero-fill
checkpoint reentrant mode
```

The remaining dominant cost is likely the finite-difference stencil update
itself during checkpoint replay.

## Next Direction

Do not continue small-allocation candidates.

The next useful path must measure or change the stencil update cost itself,
for example:

```text
1. Compare scripted production stencil against a non-scripted equivalent on the
   same NPU gate.
2. Build a focused timestep microbenchmark for the p/u/w update slices under
   checkpoint replay.
3. Only if the microbenchmark identifies a real bottleneck, test a bounded
   formula-preserving stencil rewrite.
```

Any stencil rewrite must keep receiver outputs, loss, and raw `vp.grad`
unchanged at the same fullshape gate.
