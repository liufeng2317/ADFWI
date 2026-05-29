# 105. Import Surface Policy Test

## Goal

Convert the repeated import-surface audit into a lightweight regression test so
removed compatibility shims and namespace-only aggregation imports do not
reappear in tracked code, examples, scripts, or current documentation.

## Optimization Path

1. Add `tests/test_import_surface_policy.py`.
2. Scan `git ls-files` rather than the whole working tree so ignored/generated
   notebooks and sphinx files do not create noisy failures.
3. Reject removed or namespace-only import surfaces outside historical version
   records:
   - `ADFWI.fwi.multiScaleProcessing`
   - `ADFWI.fwi.normalization`
   - `ADFWI.fwi.transforms.waveform`
   - `from ADFWI.fwi.iteration import ...`
   - `from ADFWI.fwi.runtime import ...`
4. Allow the `ADFWI.fwi.data` public facade only in the user documentation and
   the public API lock test. Framework internals and ordinary tests should keep
   using owner-module imports.

## Numerical Contract

This is a test-only import policy guard. It does not modify runtime code,
scientific helpers, tensor operations, or examples.

## Validation Result

The new policy test passed:

```bash
conda run -n adfwi python -m unittest tests/test_import_surface_policy.py
# Ran 2 tests in 2.539s, OK
```

The policy test also passed with the public facade and low-pass tests that
motivated it:

```bash
conda run -n adfwi python -m unittest tests/test_import_surface_policy.py tests/test_fwi_data_public_api.py tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py
# Ran 11 tests in 2.468s, OK
```

Syntax check passed:

```bash
conda run -n adfwi python -m py_compile tests/test_import_surface_policy.py
# OK
```

## Next Direction

Add this test to the standard lightweight validation group when doing future
compatibility cleanup.
