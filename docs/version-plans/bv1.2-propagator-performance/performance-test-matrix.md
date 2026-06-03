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

Pressure inner-state recurrence prototype:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_pressure_inner_state_microbenchmark.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 400
```

Result file:
`docs/version-plans/bv1.2-propagator-performance/acoustic_pressure_inner_state_microbenchmark_20260602.json`

| Item | Reference | Candidate |
| --- | ---: | ---: |
| recurrent state | full pressure field with sliced assignment | pressure interior only |
| output max abs diff | baseline | `0.0` |
| loss abs diff | baseline | `0.0` |
| `p/u_seq/w_seq/kappa1/alpha1` grad max abs diff | baseline | `0.0` |
| mean forward speedup | baseline | `1.1977x` |
| mean backward speedup | baseline | `1.1645x` |
| mean total speedup | baseline | `1.1714x` |

Decision: this is the first useful evidence that reducing recurrent sliced
assignment can help without gradient drift. It is still only a pressure-update
subproblem where `u/w` are provided as external time sequences, so it is not
promotion-ready. The next bounded prototype should extend the same idea to the
coupled pressure-only acoustic recurrence state (`p/u/w`) before any production
kernel edit is considered.

Coupled `p/u/w` custom-backward pressure-loss probe:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_multistep_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 3 \
  --repeat 5 \
  --steps 20 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --source-injection \
  --free-surface-boundary-write \
  --receiver-recording \
  --receivers 24 \
  --loss-kind receiver-random-linear \
  --loss-components p,rcv_p
```

Result file:
`docs/version-plans/bv1.2-propagator-performance/acoustic_custom_multistep_pressure_loss_probe_repeat_20260603.json`

| Item | Reference | Candidate |
| --- | ---: | ---: |
| recurrence | normal PyTorch autograd | custom backward for coupled `p/u/w` recurrence |
| source injection | enabled | enabled |
| free-surface boundary write | enabled | enabled |
| receiver recording | enabled | enabled |
| loss components | `p,rcv_p` | `p,rcv_p` |
| output max abs diff | baseline | `0.0` |
| loss abs diff | baseline | `0.0` |
| max gradient abs diff | baseline | `1.1920928955078125e-07` |
| max gradient rel diff | baseline | `4.812639090232551e-04` |
| mean forward speedup | baseline | `0.8779x` |
| mean backward speedup | baseline | `1.9633x` |
| mean total speedup | baseline | `1.5144x` |

Decision: this is now the strongest evidence for the high-value path. The
custom backward reduces backward cost substantially on the coupled recurrence,
with exact outputs/loss and small absolute gradient differences. It is still a
tiny benchmark, so the next step is a longer-step or production-chunk gate
before wiring anything into `acoustic_kernels.py`.

Longer-step coupled `p/u/w` pressure-loss gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_multistep_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 2 \
  --repeat 3 \
  --steps 80 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --source-injection \
  --free-surface-boundary-write \
  --receiver-recording \
  --receivers 24 \
  --loss-kind receiver-random-linear \
  --loss-components p,rcv_p
```

Result file:
`docs/version-plans/bv1.2-propagator-performance/acoustic_custom_multistep_pressure_loss_steps80_20260603.json`

| Item | Reference | Candidate |
| --- | ---: | ---: |
| recurrence steps | `80` | `80` |
| output max abs diff | baseline | `0.0` |
| loss abs diff | baseline | `0.0` |
| max gradient abs diff | baseline | `4.76837158203125e-07` |
| max gradient rel diff | baseline | `7.764155452605337e-05` |
| mean forward speedup | baseline | `0.8916x` |
| mean backward speedup | baseline | `2.0144x` |
| mean total speedup | baseline | `1.5444x` |

Decision: the custom-backward route remains stable at a longer recurrence
length. Forward is slower, but backward speedup dominates total time. The next
valid gate is a production-chunk benchmark that uses real acoustic-propagator
input preparation and compares pressure receiver output, pressure loss, and raw
`vp.grad` against the production pressure-only checkpoint path.

Production-interface chunk gates:

Observed-pressure gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --candidate-mode experimental-chunk \
  --loss-mode observed-pressure \
  --shots 3 \
  --batch-size 3 \
  --nx 64 \
  --nz 32 \
  --nt 120 \
  --checkpoint-segments 10 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination
```

Result files:

- `acoustic_production_chunk_gate_nt120_20260603.json`
- `acoustic_production_chunk_gate_nt240_20260603.json`

| Item | `nt=120` | `nt=240` |
| --- | ---: | ---: |
| output max abs diff | `0.0` | `0.0` |
| loss abs diff | `0.0` | `0.0` |
| reference raw `vp.grad` finite | `false` | `false` |
| candidate raw `vp.grad` finite | `false` | `false` |
| raw `vp.grad` diff | `NaN` | `NaN` |
| total speedup | `1.4923x` | `1.5705x` |
| candidate / reference peak memory | `3.1148x` | `8.6007x` |

Decision: this is not an acceptance gate because the production reference raw
gradient is already non-finite. It still confirms exact receiver output and
loss parity, but pressure-loss promotion needs a finite-gradient observed-data
configuration.

Synthetic-energy production-interface gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --candidate-mode experimental-chunk \
  --loss-mode synthetic-energy \
  --shots 3 \
  --batch-size 3 \
  --nx 64 \
  --nz 32 \
  --nt 120 \
  --checkpoint-segments 10 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination
```

Result file:
`acoustic_production_chunk_synthetic_energy_nt120_20260603.json`

| Item | Result |
| --- | ---: |
| output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| reference raw `vp.grad` finite | `true` |
| candidate raw `vp.grad` finite | `true` |
| raw `vp.grad` max abs diff | `7.105427357601002e-15` |
| raw `vp.grad` max rel diff | `2.2075703327573137e-06` |
| backward speedup | `1.9353x` |
| total speedup | `1.4474x` |
| candidate / reference peak memory | `5.0694x` |

Decision: the production-interface custom chunk matches raw `vp.grad` under a
finite synthetic-energy gradient gate and remains faster in backward. This is
not sufficient for production promotion because the active FWI use case is
pressure-loss inversion; next work must create a finite observed-pressure gate.

Observed-pressure finite-gradient gate:

Small-shape attempts:

| Gate | Result |
| --- | --- |
| `nx=64`, `nz=32`, `nt=120`, unnormalized observed pressure | reference and candidate raw `vp.grad` non-finite |
| `nx=64`, `nz=32`, `nt=120`, normalized observed pressure | reference and candidate raw `vp.grad` non-finite |
| production-only `nx=64`, `nz=32`, `nt=120` | production raw and processed gradients non-finite |

Decision: the small observed-pressure gate is invalid because the production
reference itself is non-finite. Do not use it for custom-backward promotion.

Reduced-shape observed-pressure production-interface gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --candidate-mode experimental-chunk \
  --loss-mode observed-pressure \
  --waveform-normalize \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --checkpoint-segments 10 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination
```

Result file:
`acoustic_production_chunk_observed_reduced_nt3000_20260603.json`

| Item | Result |
| --- | ---: |
| output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| reference raw `vp.grad` finite | `true` |
| candidate raw `vp.grad` finite | `true` |
| raw `vp.grad` max abs diff | `2.8032809495925903e-07` |
| raw `vp.grad` max rel diff | `0.15190739929676056` |
| backward speedup | `1.8051x` |
| total speedup | `1.3829x` |
| reference peak memory | `292.9844 MiB` |
| candidate peak memory | `16138.0376 MiB` |
| candidate / reference peak memory | `55.0816x` |

Decision: the production-interface custom chunk passes the finite observed
pressure-loss gate and gives meaningful speedup, but its saved-state memory is
far too high for promotion. The next valid optimization is memory reduction for
this observed-pressure production-chunk path, not more speed-only testing.

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

## Checkpoint Memory And Speed Upper Bound

This matrix compares the same reduced observed-pressure one-iteration gate
under segmented production checkpointing, production `checkpoint_segments=1`,
and the current opt-in custom chunk path. The reference is production
`checkpoint_segments=10`.

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_memory_matrix.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --loss-mode observed-pressure \
  --waveform-normalize \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --checkpoint-segments 10 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --matrix-variants production:10,production:1,production-custom-chunk:10 \
  --reference-variant production:ckpt10
```

Result file:
`acoustic_checkpoint_memory_matrix_observed_reduced_20260603.json`.

| Variant | Total time | Forward | Backward | Speedup vs ckpt10 | Peak memory | Memory vs ckpt10 | Loss diff | Raw `vp.grad` max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| production `checkpoint_segments=10` | `26.2625 s` | `3.5302 s` | `22.6946 s` | baseline | `292.9844 MiB` | baseline | baseline | baseline |
| production `checkpoint_segments=1` | `21.7277 s` | `5.7821 s` | `15.9400 s` | `1.2087x` | `2180.5459 MiB` | `7.4425x` | `0.0` | `2.2352e-08` |
| production custom chunk `checkpoint_segments=10` | `19.5471 s` | `5.9790 s` | `13.5624 s` | `1.3435x` | `7948.4272 MiB` | `27.1292x` | `0.0` | `2.7381e-07` |

Interpretation:

- `checkpoint_segments=1` is the practical no-checkpoint production upper
  bound for this gate: it is faster than checkpoint=10, but only by `1.21x`
  total while using `7.44x` peak memory. It fails the `<= 2.5x` memory
  constraint and is only a reference upper bound.
- current custom chunk is faster than checkpoint=1 in backward/total time, but
  its memory cost is much larger (`27.13x` checkpoint=10 and `3.65x`
  checkpoint=1). It also fails the `<= 2.5x` memory constraint.
- both alternatives preserve receiver output and pressure loss exactly against
  production checkpoint=10; raw `vp.grad` absolute differences remain small,
  while relative differences are inflated by near-zero gradient entries.
- the next valuable route is not more speed-only saved-state custom chunking;
  it is a memory-aware custom backward/rematerialization route that improves
  total time while staying under `2.5x` checkpoint=10 peak memory.

## Memory-Budget Remat Cache Policy Gate

This gate keeps the same reduced observed-pressure case and tests only the
rematerialized pressure-only candidate. The reference is production
`checkpoint_segments=10`; the hard budget is peak memory `<= 2.5x` reference.

| Remat cache policy | Total time | Backward | Speedup vs ckpt10 | Peak memory | Memory vs ckpt10 | Loss diff | Raw `vp.grad` max abs diff | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| cache `p,u,w` divergence every step | `23.0602 s` | `17.4832 s` | `1.2049x` | `809.5479 MiB` | `2.7631x` | `0.0` | `2.7381e-07` | fails memory budget |
| no divergence cache | `26.1894 s` | `20.7801 s` | `1.0390x` | `503.6885 MiB` | `1.7192x` | `0.0` | `2.7381e-07` | valid but weak speedup |
| cache `div_p` only every step | `24.7118 s` | `19.3851 s` | `1.0654x` | `604.5718 MiB` | `2.0635x` | `0.0` | `2.7381e-07` | current budget-valid baseline |
| cache `div_p` only, scripted pressure forward | `23.7018 s` | `20.0202 s` | `1.1752x` | `604.5718 MiB` | `2.0635x` | `0.0` | `2.7381e-07` | accepted budget-valid improvement |

Interpretation:

- state-cache stride changes are not a viable first route: `state_cache_stride=2`
  increased peak memory to `7.8443x`.
- full divergence caching is close in speed but exceeds the memory budget.
- no divergence caching is memory-safe but gives only a small total speedup.
- `div_p`-only caching is the best measured budget-valid point and should be
  the baseline for the next code-level optimization.
- scripted pressure-only remat forward removes Python list/stack overhead from
  the candidate forward pass. Forward-only timing changed from slower than
  production (`5.4453 s` candidate vs `4.0970 s` production before scripting)
  to slightly faster than production (`3.4458 s` candidate vs `3.6684 s`
  production). In the full one-iteration gate, total speedup improved from
  `1.0654x` to `1.1752x` with unchanged peak-memory ratio (`2.0635x`).

## Memory-Budget Remat Backward Stage Timing

This diagnostic enables coarse stage timing only for the current budget-valid
remat pressure path: `div_p` cache, scripted pressure forward, and
`state_cache_stride=1`. The same reduced observed-pressure case is used.

| Backward stage | Seconds | Fraction of candidate backward |
| --- | ---: | ---: |
| replay states and divergence | `5.3446 s` | `27.34%` |
| initialize gradient buffers | `0.0020 s` | `0.01%` |
| reverse adjoint loop | `14.1475 s` | `72.36%` |

Candidate timing in this diagnostic run:

| Metric | Value |
| --- | ---: |
| forward | `3.5645 s` |
| backward | `19.5506 s` |
| loss | `6375.7919921875` |
| raw `vp.grad` finite | `true` |

Interpretation:

- replay state construction is meaningful but not the dominant remaining cost;
- gradient-buffer initialization is negligible;
- the reverse adjoint loop is the next optimization target. Work should focus
  on reducing `p_new` rebuild, `div_u/div_w` recomputation, receiver adjoint
  scatter, or `_backward_step_from_saved_divergence` cost without increasing
  peak memory above `2.5x` checkpoint=10.

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
- peak memory is no more than `2.5x` the production
  `checkpoint_segments=10` baseline for the same case;
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
