# 61 - Elastic Batch Loss Step Helper

## Optimization Path

Continue from step 60 by applying the same small-batch extraction pattern to
`ElasticFWI.forward`.

The helper is scoped to one elastic batch:

- run the elastic propagator for the current `BatchRange`;
- accumulate configured elastic gradient wavefields;
- build active elastic `LossInput` records from the driver-selected components
  and component weights;
- evaluate weighted component losses through `evaluate_loss_inputs`;
- add optional model regularization;
- call the shared `apply_batch_loss_step`.

`ElasticFWI` still owns the physical component choices, anisotropic parameter
list, optimizer update, scheduler order, gradient processing, cache saving, and
figure saving.

## Change

- Added `ElasticBatchStepResult` in `ADFWI.fwi.iteration.loss`.
- Added `apply_elastic_batch_loss_step(...)` for the repeated elastic batch body.
- Updated `ElasticFWI.forward` to call the helper and keep the returned epoch
  loss scalar and accumulated wavefields.
- Added a focused iteration unit test that checks `fd_order` and checkpoint
  propagation, active component weights, pressure sign gradients for `txx/tzz`,
  `vz` gradient scaling, regularization contribution, progress label behavior,
  and pressure/vz wavefield accumulation.

## Scientific Contract

- Elastic propagator calls still receive the same `fd_order`, `shot_index`, and
  `checkpoint_segments`.
- Elastic synthetic components still come from `elastic_loss_inputs`, including
  pressure as `-(txx + tzz)`.
- Elastic observed components still use `observed_components[component][shot_index]`.
- Component order and weights still follow the driver-provided
  `inversion_component` and `component_weights`.
- Elastic gradient processing still receives the same accumulated wavefield
  selected by `select_elastic_gradient_wavefield`.
- Transform preparation, waveform normalization, misfit dispatch,
  regularization formula, backward propagation, optimizer stepping, scheduler
  stepping, model constraints, result caching, and figure saving are unchanged.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/__init__.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration.py`: 16 tests passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_runtime.py tests/test_backend_integration.py`: 69 tests passed.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, `vp_grad_norm` 0.0, `vp_update_norm` 0.0.
  - Elastic CPU/NPU relative differences: loss `4.880620563312549e-06`, `vp_grad_norm` `1.2590211728446073e-06`, `vp_update_norm` 0.0.

## Next Optimization Direction

Extract shared model-parameter specification helpers for acoustic and elastic
gradient processing. This should reduce repeated `("vp", index)` lists while
keeping the physical trainable parameter set explicit in each FWI driver.
