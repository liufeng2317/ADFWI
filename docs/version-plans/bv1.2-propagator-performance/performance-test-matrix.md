# Propagator Performance Test Matrix

Use this matrix before accepting any propagator performance change.

## Required Levels

| Level | Purpose | Required when |
| --- | --- | --- |
| Static | import and syntax safety | every propagator edit |
| Contract | wrapper and option behavior | wrapper/helper/API edits |
| Numerical parity | catch waveform/loss/gradient drift | every kernel or FWI-output edit |
| Reduced workflow | confirm real loop does not break | meaningful performance edits |
| Full-record validation | anchor major decisions | milestone changes only |

## Static And Contract Tests

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/*.py
```

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

For FWI runtime-facing output changes:

```bash
conda run -n adfwi python -m unittest \
  tests.test_fwi_runtime \
  tests.test_fwi_iteration
```

## Acoustic Parity Requirement

Run for any change to `acoustic_kernels.py`, `acoustic_propagator.py`, or
acoustic FWI output policy.

Record:

| Item | Required record |
| --- | --- |
| device/dtype | e.g. `cpu/float32`, `npu:0/float32` |
| outputs | `p`, `u`, `w`, forward wavefield tensors |
| finiteness | all checked tensors finite |
| forward difference | max abs and max relative difference |
| loss difference | absolute and relative difference |
| gradient difference | max abs and max relative `vp.grad` difference |
| timing | forward/backward/total or seconds per iteration |

Default tolerance for pure refactors:

```text
cpu float32: max_abs <= 1e-6 or max_rel <= 1e-5
npu float32: max_abs <= 1e-5 or max_rel <= 1e-4
```

Exact-code changes should aim for zero difference where the backend allows it.

## Benchmark Commands

Checkpoint/output policy probe:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 2 \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5
```

FWI iteration profile:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 10 \
  --checkpoint-segments 10
```

Current reduced checkpoint=10 baseline with AcousticFWI's auto pressure policy:

| Metric | Baseline |
| --- | ---: |
| initial loss | `6375.7919921875` |
| final loss, 10 iterations | `4776.7587890625` |
| `vp_update_norm` | `8410.6171875` |
| seconds / iteration, excluding first | `26.4793 s` |
| forward seconds / iteration, excluding first | `3.7645 s` |
| backward seconds / iteration, excluding first | `22.0680 s` |

Custom chunk speed/memory gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 5 \
  --checkpoint-segments 10 \
  --candidate-mode production-custom-chunk
```

Current result:

| Metric | Production full-output path | Custom chunk path |
| --- | ---: | ---: |
| loss trajectory | exact match | exact match |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |
| mean seconds / iteration | `27.9942 s` | `18.2167 s` |
| total speedup | baseline | `1.5367x` |
| backward speedup | baseline | `1.9362x` |
| peak allocated memory | `272.5190 MiB` | `7882.8569 MiB` |
| memory ratio | baseline | `28.9259x` |

## Efficiency Summary From Accepted Changes

The following numbers summarize accepted performance changes from
`performance-change-log.md`. They are measured on different gates and should not
be added together directly. Use the current reduced checkpoint=10 baseline above
for new comparisons.

| Change | Scope | Reduced checkpoint=10 effect | Full-record checkpoint=10 effect | Notes |
| --- | --- | ---: | ---: | --- |
| `checkpoint_segments == 1` checkpoint bypass | production kernel, no-checkpoint path | not an FWI checkpoint=10 metric | not an FWI checkpoint=10 metric | exact parity on checkpoint-overhead probe; affects `checkpoint_segments == 1` only |
| acoustic source-index hoist | production kernel cleanup | small loop cleanup, exact parity | not separately promoted as FWI-level speedup | retained as low-risk cleanup |
| skip detached illumination summaries during checkpoint replay | production checkpoint path | total `30.2897s -> 29.2227s`, `+3.52%`; backward `+4.34%` | total `28.2031s -> 27.4098s`, `+2.81%`; backward `+3.41%` | reduces useless detached illumination work during checkpoint backward replay |
| `pressure_only=True` | opt-in acoustic FWI pressure path | total `29.31s -> 25.37s`, `+13.44%` | total `28.20s -> 26.51s`, `+6.02%` | validates pressure-only path before making it AcousticFWI auto policy |
| AcousticFWI `pressure_only="auto"` | production FWI-layer policy | total `29.5813s -> 26.7011s`, `+9.74%`; backward `+9.24%` | total `28.2031s -> 25.2902s`, `+10.33%`; backward `+10.32%` | current default for AcousticFWI pressure-loss inversion loops |
| `use_custom_chunk_backward=True` | opt-in high-memory custom backward | total `139.9710s -> 91.0837s` over 5 iterations, `1.5367x`; backward `1.9362x`; loss and update exact | not yet run as full-record gate | high-value speed path, but peak allocation rises `28.9259x`; next work is memory reduction, not default promotion |

Closed candidates with measured regressions:

| Candidate | Result |
| --- | --- |
| detach-before-summary accumulation | total `+0.64%` only, forward regressed; reverted |
| empty placeholders for skipped replay summaries | reduced checkpoint=10 total regressed `-2.10%`; reverted |
| concatenate segmented receiver chunks | reduced checkpoint=10 total regressed `-1.93%`; reverted |

Pressure-only acoustic FWI opt-in comparison:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 2 \
  --checkpoint-segments 10 \
  --pressure-only
```

Required result: `p`, `forward_wavefield_p`, pressure loss, and raw `vp.grad`
match the full-output path. `u/w` receiver outputs and `u/w` wavefield summaries
are explicit zero placeholders in the pressure-only path and must not be used by
callers.

Custom chunk gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_chunk_forward.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --steps 300 \
  --shots 1 \
  --nx 64 \
  --nz 32 \
  --nabc 16 \
  --receivers 32 \
  --loss-kind receiver-random-linear \
  --loss-components rcv_p \
  --upstream-scale 4500
```

## Reduced And Full-Record Validation

Reduced inversion:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py inversion10 \
  --iterations 10 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Full-record forward:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Full 300-iteration inversion is reserved for major milestones after reduced
tests and full-record forward pass.

## Full-Record Baseline

| Metric | Baseline |
| --- | ---: |
| `record.p.shape` | `[40, 3000, 200]` |
| `record.p.norm` | `4.604166507720947` |
| forward wall time | `7.913247490301728 s` |
| initial loss | `74756.2578125` |
| final loss | `4211.63671875` |
| min loss | `2709.09228515625` |
| `vp_update_norm` | `43990.0078125` |
| seconds / iteration | `33.75309997430071 s` |

## Acceptance Rule

A performance change is acceptable only if:

- the same command, device, dtype, and case are used before and after;
- waveform/loss/gradient differences are reported;
- timing improves enough to matter for the target case;
- the result is added to `performance-change-log.md`;
- the next direction is explicitly stated.

## High-Value Adjoint Route Gate

Use this gate before any custom adjoint/backward code is wired into production.

Required prototype levels:

| Level | Purpose | Required comparison |
| --- | --- | --- |
| one-step | validate local adjoint signs and indices | custom gradient vs PyTorch autograd |
| two-step | validate state dependency through time | custom gradient vs PyTorch autograd |
| tiny multi-step | validate source injection and receiver accumulation | custom gradient vs PyTorch autograd |
| reduced FWI | validate real-loop transfer | loss trajectory, raw/processed gradients, seconds/iteration |

Required tolerances:

| Device/dtype | Receiver/loss target | Gradient target |
| --- | ---: | ---: |
| CPU float64 | max relative <= `1e-10` | max relative <= `1e-8` |
| NPU float32 | max relative <= `1e-4` | max relative <= `1e-3` |

Promotion target:

- reduced checkpoint=10 total iteration speedup should be at least `1.5x`; or
- memory reduction must be large enough to run a case that the current baseline
  cannot run, with no loss/gradient regression.
