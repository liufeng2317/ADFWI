# Acoustic Adjoint Optimization Design

This document defines the next high-value optimization route after the current
PyTorch/checkpoint cleanup phase. It is a design gate, not a production change.

## Current Baseline

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 10 \
  --checkpoint-segments 10
```

Current reduced checkpoint=10 baseline:

| Metric | Value |
| --- | ---: |
| pressure policy | `auto` |
| final loss, 10 iterations | `4776.7587890625` |
| `vp_update_norm` | `8410.6171875` |
| total seconds / iteration, excluding first | `26.4793 s` |
| forward seconds / iteration, excluding first | `3.7645 s` |
| backward seconds / iteration, excluding first | `22.0680 s` |

Interpretation:

- backward/replay is still about `83%` of the stable iteration cost;
- further small output-policy or assembly changes are unlikely to matter;
- a useful next optimization must reduce recurrence replay or replace PyTorch
  autograd for the time loop.

## Why Stop Small PyTorch Cleanup

Accepted small changes gave useful but limited gains:

- checkpoint replay illumination skip: `+3.52%` reduced, `+2.81%`
  full-record;
- AcousticFWI auto pressure policy: about `+10%` FWI-level gain.

Closed small changes show the local-cleanup limit:

- detach-before-summary accumulation: only `+0.64%`, forward regressed;
- empty replay placeholders: `-2.10%`;
- receiver chunk concatenation: `-1.93%`.

The next route should therefore target the algorithmic structure, not
expression-level TorchScript changes.

## Candidate Methods

| Method | Expected gain | Main risk | Decision |
| --- | ---: | --- | --- |
| Custom adjoint/backward for acoustic pressure FWI | high, target `1.5x+` on reduced FWI | gradient correctness and memory contract | selected |
| Fused custom NPU/CUDA kernel for time stepping | high, but hardware/toolchain heavy | NPU custom-op parity and build complexity | later, after adjoint math is validated |
| Boundary-saving reconstruction | medium/high memory benefit | reconstruction correctness and reentrancy | use only as reference material |
| JAX/XLA rewrite | potentially high on supported accelerator | uncertain NPU backend and large framework fork | not current branch |
| Numba CPU kernel | useful CPU prototype | does not accelerate current NPU target | not current branch |

## Selected Route

Build a staged acoustic pressure-only adjoint/custom-backward prototype.

Scope:

- acoustic only;
- pressure-loss FWI only;
- current second-order staggered-grid equations;
- current PML/free-surface behavior;
- start with reduced model and small synthetic tests;
- do not change `AcousticPropagator.forward` default;
- do not make the prototype default until full validation passes.

The prototype may reuse ideas from:

- `ADFWI/propagator/acoustic_custom_kernels.py`
- `ADFWI/propagator/acoustic_kernels_bs.py`

but it must not inherit their production status automatically.

## Phase Gates

### Phase 0: Baseline Lock

Use the current reduced checkpoint=10 auto-pressure baseline as the comparison
target.

Required record:

- command;
- device/dtype;
- loss trajectory;
- raw and processed gradient finiteness;
- total/forward/backward seconds per iteration;
- memory if available.

### Phase 1: One-Step And Tiny Multi-Step Parity

Goal:

Validate the local adjoint math before touching real FWI.

Required tests:

- one-step pressure receiver loss;
- two-step pressure receiver loss;
- tiny multi-step recurrence with source injection;
- CPU float64 first, then NPU float32.

Required comparisons against PyTorch autograd:

| Quantity | CPU float64 target | NPU float32 target |
| --- | ---: | ---: |
| receiver pressure | max abs <= `1e-10` | max abs <= `1e-5` |
| loss | max rel <= `1e-10` | max rel <= `1e-4` |
| `vp.grad` or coefficient grad | max rel <= `1e-8` | max rel <= `1e-3` |

Stop condition:

- any unexplained gradient mismatch;
- any nonfinite gradient;
- need for unsupported source/receiver assumptions.

### Phase 2: Chunk-Level Custom Backward

Goal:

Replace PyTorch checkpoint replay for one acoustic chunk with a custom backward
that computes the pressure-loss gradient explicitly.

Required features:

- pressure receiver output;
- source injection;
- PML damping;
- free-surface branch;
- pressure illumination output, if gradient processing needs it.

Required comparisons:

- PyTorch checkpoint path vs custom chunk path;
- receiver pressure;
- loss;
- raw `vp.grad`;
- seconds for forward/backward/total.

Promotion target:

- reduced checkpoint=10 FWI speedup >= `1.5x` on total iteration, or clear
  memory reduction with no speed regression.

### Phase 3: Reduced FWI Gate

Goal:

Run the real reduced validation loop.

Required command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 10 \
  --checkpoint-segments 10
```

Required comparisons:

- loss trajectory vs current baseline;
- raw and processed gradient finiteness;
- total/forward/backward seconds per iteration;
- `vp_update_norm`;
- output artifacts unchanged unless explicitly documented.

### Phase 4: Full-Record Gate

Goal:

Confirm that the reduced-case speedup transfers to full-record Marmousi2.

Required comparisons:

- at least 3 full-record iterations;
- final promotion only after a longer run if the change becomes default.

## Implementation Boundary

Do not:

- replace `acoustic_kernels.py` production path at the start;
- delete the PyTorch checkpoint path;
- change FWI loss semantics;
- change examples or validation outputs;
- promote custom code based only on microbenchmarks.

Do:

- add a separate prototype path;
- keep opt-in flags explicit;
- write tests before integration;
- compare raw `vp.grad` against PyTorch autograd;
- record every accepted or closed candidate in the relevant record file and add
  one-line summaries to `performance-change-log.md`.

## Next Concrete Task

Create a tiny adjoint parity test harness for acoustic pressure receiver loss.
It should run on CPU float64 first and compare custom backward gradients against
PyTorch autograd for one-step and two-step updates.
