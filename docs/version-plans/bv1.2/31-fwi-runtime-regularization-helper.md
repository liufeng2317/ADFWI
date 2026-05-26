# FWI Runtime Regularization Helper

## Goal

Continue shrinking duplicated AcousticFWI/ElasticFWI helper bodies by moving the shared single-parameter regularization loss rule into `ADFWI.fwi.runtime`, while keeping the model-specific FWI classes responsible for selecting parameters and weights.

## Change

- Added `ADFWI.fwi.runtime.calculate_regularization_loss(model_param, weight_x, weight_z, regularization_fn)`.
- Replaced the duplicated AcousticFWI/ElasticFWI method bodies with thin wrappers around the shared helper.
- Added focused runtime tests for the active, disabled-parameter, and zero-weight paths.

## Numerical Contract

The helper preserves the historical rule exactly: a parameter contributes scalar zero on its device when `requires_grad=False`; otherwise `regularization_fn.alphax` and `regularization_fn.alphaz` are set from the selected weights, and `regularization_fn.forward(model_param)` is called only when either weight is positive. Model-specific parameter ordering and total regularization accumulation remain inside AcousticFWI and ElasticFWI.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/runtime/regularization.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py tests/test_backend_integration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py` passed: 60 tests.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed.

CPU/NPU smoke comparison:

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 0.0 | pass |
| acoustic | vp_grad_norm | 0.0 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.880620563312549e-06 | pass |
| elastic | vp_grad_norm | 1.2590211728446073e-06 | pass |
| elastic | vp_update_norm | 0.0 | pass |

The smoke results match the previous bv1.2 baseline range, confirming the regularization helper extraction does not change FWI numerical behavior.
