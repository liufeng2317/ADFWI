# 03 - Utils Conversion Contract Tests

## Goal

Lock the current behavior of the lightweight conversion helpers in
`ADFWI/utils/utils.py` before changing their docstrings or implementation.

## Scope

- `tests/test_utils_conversion.py`
- `docs/version-plans/bv1.2-utils/`

No utility implementation is changed in this round.

## Covered Contracts

- `numpy2tensor` converts non-tensor inputs to torch tensors with default
  `float32` dtype and `requires_grad=False`.
- `numpy2tensor` honors explicit dtype for non-tensor inputs.
- `numpy2tensor` returns an existing tensor unchanged.
- `tensor2numpy` detaches CPU tensors and returns non-tensors unchanged.
- `gpu2cpu` converts CPU tensors with or without gradients to NumPy arrays and
  returns non-tensors unchanged.
- `list2numpy` converts Python lists and returns non-lists unchanged.
- `numpy2list` converts NumPy arrays and returns lists unchanged.

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
- `py_compile` passed for `ADFWI/utils/*.py` and the new test file.
- Only existing local NPU/Ascend warnings and a legacy low-pass tensor-copy
  warning were printed.

## Next Bounded Task

Use these tests to support a small readability pass in `ADFWI/utils/utils.py`:
clearer docstrings and internal names only, no behavior change.
