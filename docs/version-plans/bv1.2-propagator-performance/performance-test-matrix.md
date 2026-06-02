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
| seconds / iteration, excluding first | `25.9895 s` |
| forward seconds / iteration, excluding first | `3.8509 s` |
| backward seconds / iteration, excluding first | `21.4941 s` |

Current pressure-only backward operator profile:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_backward_operator_profile.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 1 \
  --checkpoint-segments 10 \
  --save-forward-wavefield \
  --pressure-only \
  --shots 1 \
  --receivers 24 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 400 \
  --topk 25
```

Result file:
`docs/version-plans/bv1.2-propagator-performance/acoustic_pressure_only_backward_operator_profile_20260602.json`

| Item | Result |
| --- | ---: |
| forward time | `0.4680 s` |
| backward time | `21.6333 s` |
| pressure record shape | `[1, 400, 24]` |
| `vp.grad` finite | `true` |
| `vp.grad` norm | `1.2599950249825298e-10` |

Top self-device events:

| Rank | Operator | Count | Self device time |
| --- | --- | ---: | ---: |
| 1 | `aten::copy_` | `31115` | `1.9107 s` |
| 2 | `aten::slice` | `56771` | `1.7485 s` |
| 3 | `CheckpointFunctionBackward` | `10` | `1.5004 s` |
| 4 | `aten::slice_backward` | `23971` | `1.4482 s` |
| 5 | `empty_tensor` | `58886` | `1.3587 s` |
| 6 | `aclnnInplaceCopy` | `30833` | `1.2739 s` |
| 7 | `autograd::engine::evaluate_function: SliceBackward0` | `23971` | `1.1574 s` |
| 8 | `aclnnInplaceZero` | `26808` | `1.1383 s` |
| 9 | `aten::zero_` | `26808` | `1.0332 s` |
| 10 | `aten::zeros` | `24406` | `1.0193 s` |

Decision: the current pressure-only checkpoint path remains dominated by
autograd bookkeeping around slicing, copying, zero allocation, and checkpoint
replay. Further Python-level receiver/output cleanup is unlikely to give a
large gain. The next valid optimization must reduce the number of autograd
slice-assignment nodes in a bounded pressure-only prototype, while preserving
loss and raw `vp.grad` parity.

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

Saved-state custom compression gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 5 \
  --checkpoint-segments 10 \
  --candidate-mode experimental-pressure-divergence-chunk
```

Current best saved-state compression point:

| Metric | Production full-output path | `div_p`-only saved-state path |
| --- | ---: | ---: |
| loss trajectory | exact match | exact match |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |
| mean seconds / iteration | `27.9868 s` | `21.2172 s` |
| total speedup | baseline | `1.3191x` |
| backward speedup | baseline | `1.5719x` |
| peak allocated memory | `272.5190 MiB` | `5643.8008 MiB` |
| memory ratio | baseline | `20.7097x` |

Candidate comparison:

| Candidate | Total speedup | Backward speedup | Peak memory | Decision |
| --- | ---: | ---: | ---: | --- |
| save no divergence | `1.2730x` | `1.4730x` | `4524.6577 MiB` | lower memory, slower |
| save only `div_p` | `1.3191x` | `1.5719x` | `5643.8008 MiB` | current best compression point |
| save only `div_u/div_w` | `1.2912x` | `1.5279x` | `6799.7402 MiB` | worse speed/memory than `div_p` only |
| save all divergence | `1.5367x` | `1.9362x` | `7882.8569 MiB` | speed ceiling, too much memory |

Rematerialized custom chunk memory-reduction gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 5 \
  --checkpoint-segments 10 \
  --candidate-mode experimental-remat-pressure-chunk \
  --remat-divergence-cache-stride 1 \
  --remat-divergence-cache-components p,u,w \
  --remat-state-cache-stride 10
```

Current pressure-loss remat result after forward-saved boundary caching,
every-step divergence caching, receiver-pressure-only recording, and skipped
velocity receiver chunk copies:

| Metric | Production full-output path | Rematerialized custom path |
| --- | ---: | ---: |
| loss trajectory | exact match | exact match |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |
| mean seconds / iteration | `26.3105 s` | `21.3357 s` |
| total speedup | baseline | `1.2332x` |
| backward speedup | baseline | `1.3994x` |
| peak allocated memory | `272.5190 MiB` | `528.1079 MiB` |
| memory ratio | baseline | `1.9379x` |

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
| lazy zero placeholders for pressure-only `u/w` outputs | production pressure-only kernel path | steady-state total `26.4793s -> 25.9895s`, `+1.85%`; backward `22.0680s -> 21.4941s`, `+2.60%`; loss and update exact | not run | removes up-front allocation of unused velocity receiver and wavefield placeholder tensors |
| `use_custom_chunk_backward=True` | opt-in high-memory custom backward | total `139.9710s -> 91.0837s` over 5 iterations, `1.5367x`; backward `1.9362x`; loss and update exact | not yet run as full-record gate | high-value speed path, but peak allocation rises `28.9259x`; next work is memory reduction, not default promotion |
| saved-state divergence compression | experimental benchmark path | best candidate saves only `div_p`: total `139.9340s -> 106.0862s`, `1.3191x`; backward `1.5719x`; loss and update exact | not run | reduces memory from saved-all `7882.8569 MiB` to `5643.8008 MiB`, but still `20.7x` production; not promotion-ready |
| rematerialized pressure-only boundary/divergence cache | experimental benchmark path | latest same-run gate total `131.5525s -> 106.6786s` over 5 iterations, `1.2332x`; backward `1.3994x`; loss and update exact | not run | keeps memory near production (`1.94x`); candidate absolute time improved from prior `111.2747s`, but still below saved-state speed ceiling |

Closed candidates with measured regressions:

| Candidate | Result |
| --- | --- |
| detach-before-summary accumulation | total `+0.64%` only, forward regressed; reverted |
| empty placeholders for skipped replay summaries | reduced checkpoint=10 total regressed `-2.10%`; reverted |
| concatenate segmented receiver chunks | reduced checkpoint=10 total regressed `-1.93%`; reverted |
| omit pressure-only velocity placeholders from internal FWI record | loss and `vp_update_norm` exact, but reduced checkpoint=10 total regressed `25.9895s -> 26.5913s`; reverted |

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
