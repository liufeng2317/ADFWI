# 04 - Utils Conversion Readability

## Goal

Make `ADFWI/utils/utils.py` easier to read after locking its current behavior
with conversion contract tests.

## Scope

- `ADFWI/utils/utils.py`
- `docs/version-plans/bv1.2-utils/`

## Changes

- Added a module docstring explaining that the helpers preserve historical ADFWI
  conversion behavior and are not a backend policy layer.
- Replaced vague docstrings with explicit contracts for tensor, NumPy, and list
  passthrough behavior.
- Renamed local parameter `a` to `value` for readability.
- Kept all conversion branches and return behavior unchanged.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_conversion.py
conda run -n adfwi python -m unittest tests/test_mute_transform_comparison.py tests/test_marmousi2_validation_example.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_conversion.py
git diff --check
```

## Result

- `tests/test_utils_conversion.py` passed: 12 tests.
- Utils-adjacent regression tests passed: 12 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_conversion.py`.
- Only existing local NPU/Ascend warnings and a legacy low-pass tensor-copy
  warning were printed.

## Next Bounded Task

Move to wavelet contract tests before changing `wavelets.py` documentation or
error messages.
