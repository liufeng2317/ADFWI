# 06 - Wavelet Documentation Correction

## Goal

Correct `ADFWI/utils/wavelets.py` documentation and typo-prone error text after
locking wavelet behavior with contract tests.

## Scope

- `ADFWI/utils/wavelets.py`
- `docs/version-plans/bv1.2-utils/`

No wavelet formula, default, return shape, or public argument name is changed.

## Changes

- Expanded the `wavelet` docstring with parameter meanings, default `t0`, and
  supported source types.
- Explicitly documented that `Gaussian` preserves the historical double
  cumulative sum of the Ricker-like base expression.
- Corrected the unsupported-type message spelling from `Rikcer`/`Guassian` to
  `Ricker`/`Gaussian`.
- Kept the public `type` argument name unchanged for compatibility.

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

## Next Bounded Task

Do not continue polishing `wavelets.py` unless a user-facing documentation issue
or numerical test gap appears. The next useful utils target is `noise.py` or
`assessment_metric.py` contract tests, because both are small and isolated.
