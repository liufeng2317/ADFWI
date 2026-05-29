# 60 - Acoustic Batch Loss Step Helper

## Optimization Path

Continue from step 59 by removing the duplicated acoustic per-batch body shared
by `AcousticFWI.forward` and `AcousticFWI.forward_closure`.

The helper is intentionally scoped to one acoustic batch:

- run the acoustic propagator for the current `BatchRange`;
- build the acoustic pressure `LossInput`;
- accumulate the pressure forward wavefield used by legacy `GradProcessor`;
- evaluate the pressure data loss through `evaluate_loss_inputs`;
- add optional model regularization;
- call the shared `apply_batch_loss_step`.

The optimizer loop, closure ownership, epoch update, scheduler order, gradient
processing, cache saving, and progress finalization remain visible in
`AcousticFWI`.

## Change

- Added `AcousticBatchStepResult` in `ADFWI.fwi.iteration.loss`.
- Added `apply_acoustic_batch_loss_step(...)` for the repeated acoustic batch
  body used by closure and non-closure optimizers.
- Updated `AcousticFWI.forward` to call the helper and keep the returned epoch
  loss scalar and accumulated wavefield.
- Updated `AcousticFWI.forward_closure` to use the same helper inside the
  optimizer closure.
- Added a focused iteration unit test that checks propagator call metadata,
  prepare-callback metadata, regularization contribution, backward gradients,
  progress label behavior, and wavefield accumulation.

## Scientific Contract

- Acoustic propagator calls still receive the same `shot_index` and
  `checkpoint_segments`.
- Acoustic synthetic pressure still comes from `record_waveform["p"]`.
- Acoustic observed pressure still uses `obs_p[shot_index]`.
- Acoustic gradient processing still receives the accumulated
  `forward_wavefield_p` NumPy array.
- Transform preparation, waveform normalization, misfit dispatch,
  regularization formula, backward propagation, optimizer stepping, scheduler
  stepping, model constraints, result caching, and figure saving are unchanged.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/__init__.py ADFWI/fwi/acoustic_fwi.py tests/test_fwi_iteration.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration.py`: 15 tests passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_runtime.py tests/test_backend_integration.py`: 69 tests passed.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, `vp_grad_norm` 0.0, `vp_update_norm` 0.0.
  - Elastic CPU/NPU relative differences: loss `4.880620563312549e-06`, `vp_grad_norm` `1.2590211728446073e-06`, `vp_update_norm` 0.0.

## Next Optimization Direction

Apply the same small-step pattern to ElasticFWI by extracting an elastic batch
loss step that preserves component selection and weighted loss visibility while
reducing the loop body around forward execution, component loss evaluation, and
wavefield accumulation.
