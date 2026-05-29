# 05 - Wavelet Contract Tests

## Goal

Lock the current behavior of `ADFWI.utils.wavelet` before changing
`wavelets.py` documentation, error messages, or implementation.

## Scope

- `tests/test_utils_wavelets.py`
- `docs/version-plans/bv1.2-utils/`

No wavelet implementation is changed in this round.

## Covered Contracts

- Time axis is `np.arange(nt) * dt`.
- Default source delay is `t0 = 1.2 / f0`.
- Wavelet type matching is case-insensitive.
- `Ricker` matches the current analytic formula.
- `Gaussian` matches the current double cumulative sum of the Ricker-like base
  formula.
- `Ramp` matches the current hyperbolic-tangent formula.
- Unknown types raise `ValueError` and include the requested type in the error.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_wavelets.py
conda run -n adfwi python -m unittest tests/test_utils_conversion.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_wavelets.py
git diff --check
```

## Result

- `tests/test_utils_wavelets.py` passed: 5 tests.
- `tests/test_utils_conversion.py` passed: 12 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_wavelets.py`.
- Only existing local NPU/Ascend warnings were printed.

## Docs Correction

The test names and this record clarify the current behavior without changing
the public function yet. Notably, `Gaussian` is the historical double cumulative
sum of the same base expression used by `Ricker`; it is not an independently
defined Gaussian pulse in this implementation.

## Next Bounded Task

If needed, update `wavelets.py` docstrings and typo-prone error text while
keeping the formulas fixed under these tests.
