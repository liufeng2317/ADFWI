# Acoustic FWI Forward-Wavefield Policy Guard

Date: 2026-05-31

## Boundary

```text
Goal:
  Expose the acoustic forward-wavefield skip option to AcousticFWI only behind
  a gradient-processor safety guard.

Scope:
  Acoustic FWI driver, acoustic batch forward helper, iteration step helper,
  and focused tests.

Validation:
  Compile, focused guard tests, processed-gradient parity test, related unit
  suite, and reduced Marmousi2 validation default path.

Stop:
  Do not enable skipped forward wavefields by default. Do not allow it when any
  active gradient processor uses forward illumination.
```

## Why This Guard Is Required

The low-level acoustic propagator can skip detached `forward_wavefield_*`
summary accumulation without changing receiver records or raw autograd
gradients. Acoustic FWI is stricter: its legacy gradient processor can use
`forward_wavefield_p` as an illumination preconditioner.

Therefore, `AcousticFWI.forward(..., save_forward_wavefield=False)` is allowed
only if every active gradient processor for trainable acoustic parameters has:

```python
forw_illumination == False
```

If a processor is missing, unknown, or has illumination enabled, the FWI driver
raises `ValueError` before the first forward pass.

## Code Change

Files:

```text
ADFWI/fwi/acoustic_fwi.py
ADFWI/fwi/runtime/forward.py
ADFWI/fwi/iteration/step.py
```

New acoustic FWI option:

```python
AcousticFWI.forward(..., save_forward_wavefield: bool = True)
AcousticFWI.forward_closure(..., save_forward_wavefield: bool = True)
```

Default behavior:

```text
save_forward_wavefield=True
  unchanged FWI behavior
  forward_wavefield_p is accumulated and passed to GradProcessor
```

Guarded opt-in behavior:

```text
save_forward_wavefield=False
  allowed only when active gradient processors have forw_illumination=False
  receiver records and pressure loss keep the normal autograd graph
  forward_wavefield_p/u/w are skipped at the propagator level
```

## Precision Contract

For the guarded opt-in path:

```text
Default path:
  unchanged output and FWI behavior.

Opt-in path with forw_illumination=False:
  loss history must match the default path;
  processed vp gradient must match the default path;
  model update must match the default path.

Opt-in path with forw_illumination=True:
  must raise before propagation.
```

This preserves the rule that performance work cannot silently change gradient
semantics.

## Static And Unit Tests

Compile:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/acoustic_fwi.py \
  ADFWI/fwi/runtime/forward.py \
  ADFWI/fwi/iteration/step.py \
  ADFWI/propagator/acoustic_kernels.py \
  ADFWI/propagator/acoustic_propagator.py
```

Focused tests:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_fwi_rejects_skipped_forward_wavefield_when_illumination_is_active \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_fwi_allows_skipped_forward_wavefield_without_illumination \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_forward_wavefield_output_can_be_skipped_without_changing_receiver_gradient \
  tests.test_fwi_runtime.FWIRuntimeTests.test_acoustic_forward_batch_keeps_shot_index_with_record \
  tests.test_fwi_iteration.TestFWIIterationHelpers.test_apply_acoustic_batch_loss_step_runs_forward_loss_backward_and_wavefield_accumulation
```

Result:

```text
Ran 5 tests in 0.479s
OK
```

Processed-gradient parity test:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_fwi_skipped_forward_wavefield_matches_default_without_illumination
```

Result:

```text
Ran 1 test in 11.068s
OK
```

Full related unit suite:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_fwi_runtime \
  tests.test_fwi_iteration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

Result:

```text
Ran 79 tests in 12.041s
OK
```

## Reduced Validation Default Path

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 2 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/fwi_wavefield_guard_default_validation
```

Result:

```text
status: ok
forward record.p.shape: [3, 3000, 200]
forward record.p.norm: 1.3137102127075195
forward seconds: 6.749620955437422
inversion initial_loss: 6375.7919921875
inversion final_loss: 6006.28125
vp_update_norm: 2295.9599609375
```

This confirms the default FWI path remains aligned with the existing reduced
validation baseline.

## Remaining Risk

- The guarded opt-in path has not been run on the full Marmousi2 validation
  case because the current validation scripts intentionally use the legacy
  `GradProcessor` default, where `forw_illumination=True`.
- The guard is acoustic-only. Elastic kernels and elastic FWI still use their
  existing forward-wavefield behavior.
- Custom gradient processors without a clear `forw_illumination=False` marker
  are treated as unsafe.

## Next Direction

If this optimization is needed in real FWI scripts, add an explicit validation
case where the user intentionally sets `GradProcessor(forw_illumination=False)`
and passes `save_forward_wavefield=False`. Then compare reduced inversion loss,
processed gradients, and model updates against the same case with default
forward-wavefield accumulation.
