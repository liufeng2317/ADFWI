# 63 - FWI Model Regularization Sum Helper

## Optimization Path

Continue from step 62 by moving the repeated ordered model-parameter
regularization summation out of AcousticFWI and ElasticFWI.

The helper keeps parameter ownership explicit:

- AcousticFWI passes `acoustic_parameter_names()`;
- ElasticFWI passes `elastic_parameter_names(include_anisotropic=...)`;
- weight lists are indexed in that same order;
- per-parameter regularization behavior remains delegated to the existing
  `calculate_regularization_loss(...)` helper.

## Change

- Added `calculate_model_regularization_loss(...)` in
  `ADFWI.fwi.runtime.regularization`.
- Exported the helper through `ADFWI.fwi.runtime`.
- Updated `AcousticFWI.calculate_model_regularization_loss` to delegate ordered
  vp/rho summation to the helper.
- Updated `ElasticFWI.calculate_model_regularization_loss` to delegate ordered
  isotropic/anisotropic parameter summation to the helper.
- Added a focused runtime test for ordered weight indexing, trainable-parameter
  summation, and frozen-parameter zero contribution.

## Scientific Contract

- Acoustic regularization still uses `vp` then `rho` with the same x/z weight
  indices.
- Elastic regularization still uses `vp`, `vs`, `rho`, and for anisotropic
  models `eps`, `delta`, `gamma`, in the same order as before.
- Disabled parameters still contribute a scalar zero on their device.
- Parameters with zero x/z weights still skip the regularization forward call.
- Regularization formulas, loss construction, backward propagation, gradient
  processing, optimizer stepping, scheduler stepping, model constraints, and
  cache behavior are unchanged.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/runtime/regularization.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py`: 24 tests passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_iteration.py tests/test_backend_integration.py`: 65 tests passed.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, `vp_grad_norm` 0.0, `vp_update_norm` 0.0.
  - Elastic CPU/NPU relative differences: loss `4.880620563312549e-06`, `vp_grad_norm` `1.2590211728446073e-06`, `vp_update_norm` 0.0.

## Next Optimization Direction

Clean up legacy multiscale low-pass compatibility documentation and comments.
The implementation already lives in `ADFWI.fwi.multiscale.legacy_lowpass` while
`ADFWI.fwi.multiScaleProcessing` re-exports it; the next low-risk step is to
make the ownership, CPU/NPU limitation, and migration path clearer without
changing numerical behavior.
