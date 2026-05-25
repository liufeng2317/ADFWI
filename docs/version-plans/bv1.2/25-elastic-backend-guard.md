# Elastic Backend Guard

## Goal

Bring ElasticFWI to the same backend safety level as AcousticFWI before deeper loop cleanup. Elastic inversion should fail early when model and propagator devices disagree, and regularization tensors should follow the propagator backend automatically.

## Change

- Added `ElasticFWI._validate_device_consistency`.
- Added `ElasticFWI._align_regularization_backend`.
- Called both helpers during ElasticFWI initialization after `self.device` and `self.dtype` are derived from the propagator.
- Added backend integration tests for elastic regularization alignment and device mismatch rejection.

## Numerical Contract

This pass does not change propagation, data transforms, misfit evaluation, or regularization formulas. It only aligns backend metadata/tensors at construction time and adds an early error for inconsistent model/propagator devices. Existing CPU/NPU smoke values should remain unchanged within the established tolerance.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/elastic_fwi.py tests/test_backend_integration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py` passed: 53 tests.
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

The smoke results match the previous bv1.2 baseline range, confirming the guard/alignment pass does not change FWI numerical behavior.
