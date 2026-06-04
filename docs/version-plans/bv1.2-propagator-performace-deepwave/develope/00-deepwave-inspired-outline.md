# Deepwave-Inspired Optimization Outline

## 1. Why This Direction

The previous bv1.2 propagator-performance work showed that small Python-level
optimizations are now close to exhaustion:

- pressure-only FWI and skipped unused wavefield summaries give useful
  production gains;
- opt-in pressure rematerialization can improve iteration speed, but memory and
  complexity are difficult to control;
- repeated edits to Python slicing, receiver collection, or local clones give
  small/noisy improvements and are easy to overfit.

Deepwave uses a different architecture:

```text
User API
  |
  v
Python validation and tensor preparation
  |
  v
torch.autograd.Function boundary
  |
  v
C/CUDA forward kernel
  |
  v
explicit C/CUDA adjoint/backward kernel
  |
  v
controlled intermediate storage policy
```

The key lesson is not a single trick. The key lesson is to make propagation a
single differentiable operator with explicit forward/backward/storage contracts,
instead of letting PyTorch build a very large Python time-loop autograd graph.

## 2. Optimization Target

The first target should be the acoustic pressure-loss FWI path, because it is
the dominant real case and has the clearest validation baseline.

Baseline reference:

- full-record Marmousi2 acoustic validation case;
- 40 shots, 10 iterations, existing bv1.2 production path;
- checkpoint behavior recorded in the current performance matrix;
- loss trajectory, raw/processed gradients, and memory/iteration time must be
  compared.

## 3. Top-To-Bottom Plan

```text
Phase A: Contract Definition
  A1. Define exact acoustic operator inputs/outputs
  A2. Define gradients required by FWI
  A3. Define storage modes and memory budget
  A4. Define validation gates

Phase B: Minimal Custom Operator Prototype
  B1. Keep current Python propagator unchanged
  B2. Build a separate acoustic operator prototype
  B3. Implement forward only first
  B4. Compare receiver outputs against ADFWI production

Phase C: Explicit Adjoint Prototype
  C1. Implement backward/adjoint for pressure-loss path
  C2. Compare raw vp gradient against production autograd
  C3. Confirm finite gradients and loss trajectory in short FWI

Phase D: Storage Policy Prototype
  D1. Add device storage mode
  D2. Add rematerialized/block storage mode
  D3. Add optional CPU/disk/compressed storage only if needed
  D4. Compare speed-memory curve against checkpoints=10 and checkpoints=1

Phase E: Production Integration
  E1. Add opt-in API only
  E2. Keep default production path unchanged
  E3. Run full 40-shot 10-iteration validation
  E4. Promote only if speed gain and numerical gates pass
```

## 4. What Not To Do

- Do not continue adding more Python remat variants without a storage contract.
- Do not make a custom path default before full FWI validation.
- Do not optimize receiver output or plotting paths as the main performance
  route.
- Do not copy Deepwave source into ADFWI. Use ideas and design principles only.
- Do not mix acoustic and elastic optimization in the first implementation.

## 5. Proposed Success Gates

Minimum acceptance for an opt-in acoustic custom operator:

| Gate | Requirement |
| --- | --- |
| Receiver output | max absolute difference within established float32 tolerance |
| Loss trajectory | identical or tolerance-bounded for 5/10 iteration FWI |
| Raw vp gradient | finite and tolerance-bounded against production |
| Processed gradient/update | update norm matches production within tolerance |
| Speed | at least 1.3x on a representative FWI loop, or clear path to 1.5x |
| Memory | configurable; at least one mode should stay near checkpoint=10 memory |
| API | opt-in only, no default behavior change |

## 6. Recommended First Implementation Route

Start with a narrow acoustic scalar operator prototype:

1. create a separate prototype module, not another branch of
   `acoustic_kernels.py`;
2. expose only the pressure receiver path first;
3. keep PML/source/receiver indexing identical to current ADFWI;
4. implement forward parity before any backward work;
5. implement explicit adjoint only after forward parity is stable;
6. record every result in the test matrix, not in many per-run documents.

This route is larger than a micro-optimization, but it is the first route that
can plausibly change the performance ceiling.

