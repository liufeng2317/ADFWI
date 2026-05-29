# 01 - Utils Planning Audit

## Goal

Start `ADFWI/utils` cleanup by recording the current responsibilities and direct
validation surface before changing helper code.

## Scope

- `ADFWI/utils/`
- direct callers from examples, validation scripts, transforms, propagators, and
  tests;
- `docs/version-plans/bv1.2-utils/`.

No utility function implementation is changed in this round.

## Findings

- `ADFWI/utils` is a shared helper layer, not a single scientific module.
- Current helpers fall into five practical groups:
  1. array/list/tensor conversion;
  2. source wavelet generation;
  3. benchmark dataset/model helpers;
  4. legacy waveform mute and signal-processing helpers;
  5. metrics and noise helpers.
- `velocityDemo.py` is the least cleanly named/organized file, but it feeds
  existing examples and validation scripts, so it should not be moved before
  dataset-helper validation exists.
- `first_arrivel_picking.py` has a misspelled filename, but it is part of the
  legacy mute compatibility path and should not be renamed in a casual cleanup.
- `__init__.py` has a broad public surface and a duplicate
  `get_linear_vel_model` export. This is a candidate for the first code cleanup
  round, but only after import smoke validation.

## Validation

```bash
conda run -n adfwi python -m py_compile ADFWI/utils/*.py
conda run -n adfwi python - <<'PY'
from ADFWI.utils import wavelet, numpy2tensor, load_marmousi_model, resample_marmousi_model, mute_offset
print("utils import ok")
PY
conda run -n adfwi python -m unittest tests/test_mute_transform_comparison.py tests/test_marmousi2_validation_example.py
git diff --check
```

## Result

- `py_compile` passed for all files in `ADFWI/utils`.
- Public import smoke passed.
- Utils-adjacent tests passed: 12 tests.
- Only existing local NPU/Ascend warnings and a legacy low-pass tensor-copy
  warning were printed.

## Next Bounded Task

Run a minimal package readability pass:

- add a package docstring to `ADFWI/utils/__init__.py`;
- group exports by responsibility;
- remove the duplicate `get_linear_vel_model` export if import smoke and tests
  remain unchanged.

Stop there. Do not enter `velocityDemo.py` internals until conversion and
wavelet contracts have focused tests.

