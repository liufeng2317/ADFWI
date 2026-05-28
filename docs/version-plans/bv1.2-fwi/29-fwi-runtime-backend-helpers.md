# FWI Runtime Backend Helpers

## Goal

Remove duplicated AcousticFWI/ElasticFWI backend guard logic while keeping backend/device behavior explicit. This continues the bv1.2 organization pass by putting runtime construction helpers into a small `ADFWI.fwi.runtime` package instead of growing flat helper files under `ADFWI.fwi`.

## Change

- Added `ADFWI.fwi.runtime.validate_model_propagator_devices(model, propagator)`.
- Added `ADFWI.fwi.runtime.align_regularization_backend(regularization_fn, device, dtype)`.
- Replaced duplicated private methods in `AcousticFWI` and `ElasticFWI` with calls to the shared helpers.
- Added focused runtime helper tests for device mismatch errors, regularization tensor dtype/device alignment, and the `None` no-op path.

## Numerical Contract

This pass does not change propagation, transform order, misfit dispatch, regularization formulas, optimizer behavior, or batch logic. It only moves constructor-time backend checks and regularization tensor alignment into shared helpers. Existing CPU/NPU smoke values should remain unchanged within the established tolerance.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/runtime/__init__.py ADFWI/fwi/runtime/backend.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py tests/test_backend_integration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py` passed: 57 tests.
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

The smoke results match the previous bv1.2 baseline range, confirming the runtime helper extraction does not change FWI numerical behavior.
