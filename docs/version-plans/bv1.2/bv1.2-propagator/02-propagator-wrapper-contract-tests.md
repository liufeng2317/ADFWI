# 02 - Propagator Wrapper Contract Tests

## Goal

Add focused construction tests for acoustic and elastic propagator wrappers
without changing propagator implementation.

## Scope

- `tests/test_backend_integration.py`
- `docs/version-plans/bv1.2-propagator/`

## Optimization Path

1. Add acoustic wrapper construction contract coverage:
   - source/receiver tensor shape;
   - source/receiver index dtype;
   - wavelet dtype/device;
   - moment tensor shape;
   - receiver mask is stored on the wrapper;
   - `receiver_masks_obs` is preserved.
2. Add elastic wrapper construction contract coverage with the same survey-side
   checks.
3. Keep tests at construction level only. Do not call `forward()` and do not
   exercise finite-difference kernels.

## Validation

This round adds tests only. It does not change kernel code, boundary formulas,
checkpoint behavior, or wrapper implementation:

- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`
- `conda run -n adfwi python -m py_compile tests/test_backend_integration.py`
- `git diff --check`

## Result

- `conda run -n adfwi python -m unittest tests/test_backend_integration.py` 通过，24 tests OK。
- `conda run -n adfwi python -m py_compile tests/test_backend_integration.py` 通过。
- `git diff --check` 通过。

本轮结论：

- Acoustic wrapper construction contract now covers source/receiver tensor
  shapes, long index dtype, wavelet dtype/device, moment tensor shape, and
  receiver mask metadata preservation.
- Elastic wrapper construction contract now covers the same survey-side tensor
  and receiver mask metadata behavior.
- No propagator implementation or numerical forward path was changed.

## Stop Rule

No propagator implementation edits, no forward numerical run, no kernel,
boundary-condition, checkpoint, or gradient-processing changes.
