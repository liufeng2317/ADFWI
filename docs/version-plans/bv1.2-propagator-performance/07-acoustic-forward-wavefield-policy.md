# Acoustic Forward-Wavefield Policy Prototype

Date: 2026-05-31

## Boundary

```text
Goal:
  Add an opt-in acoustic path that skips detached forward-wavefield summary
  accumulation while preserving receiver records and autograd gradients.

Scope:
  Acoustic propagator and acoustic kernel only.

Validation:
  Compile, unit contract test, NPU default-path numerical parity, NPU opt-in
  receiver/loss/vp-gradient parity, reduced validation, and full-record forward.

Stop:
  Do not connect this option to FWI iteration yet. Do not use it when gradient
  processing requires forward illumination.
```

## Why This Is Not A Default FWI Optimization

`forward_wavefield_p` is detached and does not participate in the autograd
misfit graph. However, acoustic FWI still accumulates it and passes it into
`GradProcessor` as the `forw` illumination input:

```text
propagator.forward
  -> record["forward_wavefield_p"]
  -> accumulate_wavefield(...)
  -> process_named_parameter_gradients(..., forw=forw)
  -> GradProcessor.forward(..., forw=forw)
```

Therefore, skipping forward-wavefield accumulation is only safe for paths that
do not need illumination preconditioning, such as forward modeling,
visualization-free benchmarks, or a future FWI path that explicitly verifies
`forw_illumination=False`.

This round only exposes the low-level opt-in option and validates that receiver
records and raw autograd gradients are unchanged. It does not change default
FWI behavior.

## Code Change

Files:

```text
ADFWI/propagator/acoustic_kernels.py
ADFWI/propagator/acoustic_propagator.py
```

New option:

```python
AcousticPropagator.forward(..., save_forward_wavefield: bool = True)
```

Default behavior:

```text
save_forward_wavefield=True
  identical default output contract
  returns p/u/w and forward_wavefield_p/u/w as before
```

Opt-in behavior:

```text
save_forward_wavefield=False
  still returns p/u/w receiver records
  still preserves receiver waveform autograd graph
  forward_wavefield_p/u/w are returned as zero tensors
```

The finite-difference update, source injection, receiver sampling, loss path,
and checkpoint policy are unchanged.

## Safety Contract

For `save_forward_wavefield=False` to be accepted as a low-level option:

```text
receiver p/u/w diff = 0.0
pressure loss diff = 0.0
raw autograd vp.grad diff = 0.0
forward_wavefield_* are intentionally zero in skipped mode
```

This contract does not say that processed FWI gradients are unchanged when
`GradProcessor(forw_illumination=True)` is used. In that case the option must
remain disabled.

## Unit/Static Tests

Compile:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/propagator/acoustic_kernels.py \
  ADFWI/propagator/acoustic_propagator.py \
  scripts/benchmark/acoustic_wavefield_policy.py
```

Contract test:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_forward_wavefield_output_can_be_skipped_without_changing_receiver_gradient
```

Result:

```text
Ran 1 test in 0.297s
OK
```

Full related unit suite:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

Result:

```text
Ran 36 tests in 11.751s
OK
```

## Default Path NPU Parity

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 2 \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_wavefield_policy_default_parity_20260531.json
```

Result:

```text
loss_abs_diff: 0.0
loss_rel_diff: 0.0
p/u/w max_abs_diff: 0.0
p/u/w max_rel_diff: 0.0
forward_wavefield_p/u/w max_abs_diff: 0.0
forward_wavefield_p/u/w max_rel_diff: 0.0
vp_grad max_abs_diff: 0.0
vp_grad max_rel_diff: 0.0
```

Default timing summary:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| default implemented path | 1.6522 | 4.8784 | 6.5305 |
| direct monkeypatch | 1.6465 | 4.8996 | 6.5461 |

## Opt-In Skip NPU Parity

Benchmark script:

```text
scripts/benchmark/acoustic_wavefield_policy.py
```

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_wavefield_policy.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_wavefield_policy_skip_compare_20260531.json
```

Result across all three repeats:

```text
loss_abs_diff: 0.0
p/u/w max_abs_diff: 0.0
p/u/w max_rel_diff: 0.0
vp_grad max_abs_diff: 0.0
vp_grad max_rel_diff: 0.0
forward_wavefield_p/u/w skipped norms: 0.0
```

Timing:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| `save_forward_wavefield=True` | 1.6388 | 4.8536 | 6.4924 |
| `save_forward_wavefield=False` | 1.4209 | 4.7243 | 6.1452 |

Observed speedup on this small NPU benchmark:

```text
forward: about 1.15x
total: about 1.06x
```

This speedup is valid only for opt-in paths where the detached forward
wavefield summaries are not needed.

## Reduced Validation Default Path

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 2 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/wavefield_policy_default_validation
```

Result:

```text
status: ok
forward record.p.shape: [3, 3000, 200]
forward record.p.norm: 1.3137102127075195
forward seconds: 6.883428253233433
inversion initial_loss: 6375.7919921875
inversion final_loss: 6006.28125
vp_update_norm: 2295.9599609375
```

## Full-Record Forward Default Path

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_full_record/outputs/wavefield_policy_default_full_forward
```

Result:

```text
status: ok
record.p.shape: [40, 3000, 200]
record.p.norm: 4.604166507720947
record.p.finite: true
seconds: 7.394755927845836
```

The norm matches the full-record baseline exactly.

## Remaining Risk

- This option must not be used in normal acoustic FWI while
  `GradProcessor.forw_illumination=True`.
- The option returns zero `forward_wavefield_*` tensors instead of removing
  keys, preserving the output dictionary shape contract but making the skipped
  state explicit through zero norms.
- A full 300-iteration inversion with this option is intentionally not run
  because it is not yet connected to a safe FWI policy.

## Next Direction

Do not connect `save_forward_wavefield=False` to FWI globally. The safe next
step is to add an FWI-level guard that only allows this option when all active
gradient processors have `forw_illumination=False`, then validate processed
gradients and reduced inversion behavior against the default path.
