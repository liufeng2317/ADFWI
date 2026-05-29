# 02 - Utils Public Namespace Readability

## Goal

Clarify the public `ADFWI.utils` package namespace without changing helper
implementations or numerical behavior.

## Scope

- `ADFWI/utils/__init__.py`
- `docs/version-plans/bv1.2-utils/`

No changes are made to conversion helpers, wavelet formulas, model loaders,
mute helpers, metrics, noise, or frequency-processing functions.

## Changes

- Added a package docstring explaining that `ADFWI.utils` is a broad
  compatibility namespace for examples and framework glue.
- Grouped package-level imports by responsibility.
- Removed one duplicate `get_linear_vel_model` entry from the import list.
- Kept the historical broad `frequency_domin_process` re-export for existing
  notebooks and scripts.

## Validation

```bash
conda run -n adfwi python -m py_compile ADFWI/utils/*.py
conda run -n adfwi python - <<'PY'
from ADFWI.utils import wavelet, numpy2tensor, load_marmousi_model, resample_marmousi_model, mute_offset, get_linear_vel_model
print("utils import ok")
PY
conda run -n adfwi python -m unittest tests/test_mute_transform_comparison.py tests/test_marmousi2_validation_example.py
git diff --check
```

## Result

- `py_compile` passed for all files in `ADFWI/utils`.
- Public import smoke passed, including `wavelet`, conversion helpers,
  Marmousi helpers, `mute_offset`, and `get_linear_vel_model`.
- Utils-adjacent tests passed: 12 tests.
- Only existing local NPU/Ascend warnings and a legacy low-pass tensor-copy
  warning were printed.

## Next Bounded Task

Add focused tests for `utils.py` conversion helpers before changing any
conversion docstrings or internals.
