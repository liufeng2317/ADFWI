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

## Current Custom-Backward Baseline

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 5 \
  --checkpoint-segments 10 \
  --candidate-mode production-custom-chunk \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_custom_chunk_fwi_loop_compare_20260602.json
```

Result:

| Metric | Production full-output path | `use_custom_chunk_backward=True` |
| --- | ---: | ---: |
| loss trajectory | exact match | exact match |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |
| total seconds, 5 iterations | `139.9710 s` | `91.0837 s` |
| mean seconds / iteration | `27.9942 s` | `18.2167 s` |
| forward seconds | `20.2829 s` | `29.6301 s` |
| backward seconds | `118.8969 s` | `61.4076 s` |
| peak allocated memory | `272.5190 MiB` | `7882.8569 MiB` |

Derived:

- total loop speedup: `1.5367x`;
- backward speedup: `1.9362x`;
- forward path slows down to `0.6845x`;
- peak allocation increases by `28.9259x`;
- loss and final model update are identical for this 5-iteration gate.

Interpretation:

- the custom chunk backward is the current measured speed ceiling inside
  Python/Torch production integration;
- its speedup is real enough to continue;
- the memory increase is too large for default promotion;
- the next useful work is not another micro-optimization, but reducing saved
  custom-backward state while preserving the measured backward benefit.

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

Continue the acoustic custom-backward line, but shift the objective from
proving speed to reducing memory.

Scope:

- acoustic only;
- pressure-loss FWI only;
- current second-order staggered-grid equations;
- current PML/free-surface behavior;
- reduced FWI timing remains the main gate;
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

### Phase 1: Local Parity Gate

Goal:

Keep local adjoint tests as the safety gate for any change to custom backward
state storage.

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

### Phase 2: Memory-Reduced Custom Backward

Goal:

Reduce saved custom-backward state without giving away the backward speedup
already measured by `use_custom_chunk_backward=True`.

Required features:

- pressure receiver output;
- source injection;
- PML damping;
- free-surface branch;
- pressure illumination output, if gradient processing needs it.

Required comparisons:

- current saved-state custom chunk path vs memory-reduced candidate;
- PyTorch checkpoint path as the scientific reference;
- receiver pressure;
- loss;
- raw `vp.grad`;
- peak allocated memory;
- seconds for forward/backward/total.

Promotion target:

- loss trajectory and `vp_update_norm` match the production reference;
- peak allocation is materially below the current `7882.8569 MiB`;
- total iteration remains meaningfully faster than the production full-output
  path;
- if the candidate cannot beat the current saved-state path on memory, stop
  this line.

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

Design and test one memory-reduced custom-backward candidate against the current
saved-state custom chunk baseline:

1. keep production `acoustic_kernels.py` unchanged;
2. work only in `acoustic_custom_kernels.py` and benchmark scripts;
3. compare production full-output, current saved-state custom chunk, and the
   memory-reduced candidate on reduced FWI;
4. report loss trajectory, `vp_update_norm`, gradient finiteness, peak memory,
   forward/backward/total timing;
5. continue only if memory drops substantially without losing the core backward
   speed benefit.
