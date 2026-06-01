# Fused Acoustic Stencil Feasibility

Date: 2026-06-01

## Purpose

This document defines the next acoustic propagator performance direction after
closing the Python-level optimization lines.

Previous results showed:

```text
Python/TorchScript expression rewrites: too small, <= 1.017x
production small-allocation candidates: too small, about 1.01x-1.03x
rematerialized Python custom chunk: numerically safe but only 1.087x in 5-iter FWI with 2.266x memory
high-memory custom chunk: useful opt-in speed mode, but not checkpoint memory equivalent
```

The remaining measured cost is the finite-difference stencil itself and its
autograd replay/slice-copy overhead. Therefore the only acoustic-kernel route
with plausible larger speedup is a lower-level fused stencil prototype.

This is a feasibility contract only. No production kernel is changed in this
round.

## Target

The target is the acoustic timestep recurrence in:

```text
ADFWI/propagator/acoustic_kernels.py
```

Specifically:

```text
p update
u update
w update
source injection
free-surface handling
receiver sampling
```

The first fused prototype must stay outside the default production path. A
candidate location can be:

```text
scripts/benchmark/
ADFWI/propagator/acoustic_fused_stencil_experimental.py
```

but it must not replace `forward_kernel` until all gates pass.

## What "Fused Stencil" Means Here

The goal is not to rewrite formulas in Python. That path is already closed.

Here "fused" means reducing the number of high-level tensor slice operations and
autograd nodes created per timestep. Possible implementations include:

```text
1. a lower-level custom operator;
2. a backend-specific extension if available;
3. a TorchScript/custom-autograd block that materially reduces graph nodes;
4. a boundary-saving stencil prototype with an explicit adjoint contract.
```

If the available environment cannot support a lower-level fused implementation,
this line should stop rather than continue Python micro-optimizations.

## Non-Negotiable Scientific Contract

The fused prototype must preserve the same discrete acoustic update.

It must not change:

```text
finite-difference coefficients c1=9/8 and c2=-1/24
update order: p -> source/free surface -> u -> w -> receiver
PML damping formulas
free-surface semantics
source injection semantics
receiver sampling semantics
dtype/device behavior
default public output keys
```

Gradient behavior is part of the contract. A faster forward path that changes
raw `vp.grad` is not acceptable.

## Feasibility Gate 0: Implementation Path Check

Before writing a fused prototype, record which implementation path is actually
available in this environment:

| Path | Check | Stop condition |
| --- | --- | --- |
| C++/CUDA/NPU extension | Can it build and run in `adfwi`? | stop if toolchain or backend API is unavailable |
| `torch.compile`/graph capture | Does it support the NPU stack? | stop if unsupported or slower, as earlier probes suggested |
| custom autograd Python block | Can it reduce graph nodes without huge memory? | stop if it repeats the remat/custom-chunk tradeoff |
| boundary-saving prototype | Can it define a clear adjoint contract? | keep separate from production until gradient gates pass |

This gate prevents starting implementation work without a viable execution
backend.

## Feasibility Gate 1: One-Step Stencil Parity

Smallest numerical gate:

```text
case: synthetic one-step acoustic state
device: cpu first, then npu:0
dtype: float64 on CPU when possible, float32 on NPU
shape: small enough for local debugging
```

Required comparison against the production sliced update:

| Quantity | Requirement |
| --- | --- |
| `p/u/w` output | max abs and max rel diff recorded |
| loss from `p/u/w` | max abs diff recorded |
| gradients wrt `p/u/w` | finite and within tolerance |
| gradients wrt `kappa/alpha` or `vp/rho` equivalent | finite and within tolerance |

Acceptance:

```text
CPU float64: near machine precision for formula-level tensors
NPU float32: no larger than existing accepted acoustic custom-gradient parity scale
```

## Feasibility Gate 2: Multi-Step Recurrence Parity

Run a short recurrence before connecting to validation geometry:

```text
timesteps: 20, 100, then 300
sources: 1 and batched 3
free_surface: true and false if practical
losses: synthetic energy and receiver pressure
```

Required comparison:

```text
final p/u/w
receiver p/u/w
loss
raw input-state gradients
finite checks
forward time
backward time
peak memory
```

Stop if gradient differences grow with time in a way that cannot be explained
by normal float32 accumulation.

## Feasibility Gate 3: Validation-Geometry One Iteration

Use the reduced Marmousi2 fullshape gate:

```text
validation-case: reduced
device: npu:0
dtype: float32
checkpoint_segments: 10
shots: 3
batch_size: 3
nx/nz/nt: 200/88/3000
save_forward_wavefield: false
grad_forw_illumination: false
loss: observed pressure with waveform normalization
```

Required comparison against production:

| Quantity | Requirement |
| --- | --- |
| receiver `p/u/w` | max abs and rel diff recorded |
| loss | abs diff recorded |
| raw `vp.grad` | max abs and rel diff recorded |
| raw/processed gradient finite | must pass |
| timing | forward/backward/total |
| memory | peak allocated |

Acceptance for continuing:

```text
total one-iteration speedup >= 1.15x
raw vp.grad difference no worse than accepted custom-chunk scale
peak memory not worse than the high-memory custom chunk unless explicitly labeled high-memory
```

## Feasibility Gate 4: Short Real FWI

Only run this if Gate 3 passes.

Required run:

```text
5 or 10 iterations
same reduced fullshape Marmousi2 setup
production vs fused prototype
```

Required comparison:

```text
loss trajectory
raw gradient finite per iteration
processed gradient finite per iteration
vp_update_norm
seconds / iteration
peak memory
```

Acceptance:

```text
loss trajectory unchanged within established float32 tolerance
total loop speedup >= 1.15x
memory position clearly stated: default-safe, opt-in high-memory, or rejected
```

## Stop Conditions

Stop this line immediately if:

```text
1. no viable lower-level implementation path exists;
2. the fused prototype cannot preserve one-step gradient parity;
3. multi-step gradient differences grow beyond accepted tolerance;
4. validation one-iteration speedup is below 1.15x;
5. short FWI speedup falls below 1.15x;
6. the only available implementation is another Python slice rearrangement.
```

## First Concrete Task

Do not edit `ADFWI/propagator/acoustic_kernels.py`.

The next task is an implementation-path probe:

```text
Check whether the current adfwi environment can support a lower-level fused
operator path on the active NPU stack. If not, document the blocker and stop
the fused-stencil line.
```

If a lower-level path is available, then create a one-step prototype benchmark
outside production. If it is not available, the correct decision is to stop
acoustic kernel performance work here or move to boundary-saving research as a
separate project.
