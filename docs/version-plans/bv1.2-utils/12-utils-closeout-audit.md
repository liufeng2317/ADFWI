# 12 - Utils Closeout Audit

## Goal

Close the current `ADFWI/utils` cleanup stage by recording what is already
covered and what should remain as known risk, without starting another
open-ended readability pass.

## Scope

- `ADFWI/utils/`
- utils-focused tests under `tests/`
- `docs/version-plans/bv1.2-utils/`

No implementation code is changed in this round.

## Current State

The utils package is now documented as a compatibility helper layer rather than
one coherent numerical module. Its covered areas are:

- tensor/list/NumPy conversion helpers;
- source wavelet generation;
- Gaussian noise and model-quality metrics;
- frequency spectrum/filter helper behavior;
- legacy first-arrival and offset mute helpers.

The public package surface remains intentionally broad because many examples
still use `from ADFWI.utils import *` or import model helpers directly from the
package namespace.

## Validation

```bash
conda run -n adfwi python -m py_compile ADFWI/utils/*.py
conda run -n adfwi python -m unittest \
  tests/test_utils_conversion.py \
  tests/test_utils_wavelets.py \
  tests/test_utils_noise_metrics.py \
  tests/test_utils_frequency_process.py \
  tests/test_utils_legacy_mute.py \
  tests/test_mute_transform_comparison.py
conda run -n adfwi python -c "import scipy, ADFWI.utils as utils; from scipy.interpolate import interp2d; print(scipy.__version__, hasattr(utils, 'wavelet'))"
```

## Result

- `py_compile` passed for all files in `ADFWI/utils`.
- Utils-focused tests passed: 37 tests.
- `ADFWI.utils` import smoke passed in the `adfwi` environment.
- Current SciPy version is `1.13.1`; `scipy.interpolate.interp2d` is still
  importable in this environment.
- Only existing local NPU/Ascend warnings, Matplotlib/PyParsing deprecation
  warnings, and the legacy low-pass tensor-copy warning were printed.

## Remaining Risk

1. `velocityDemo.py` is still the largest compatibility risk.
   It mixes dataset download, file-format loading, model construction,
   smoothing, resampling, and legacy example support in one file. This is not
   ideal, but changing it requires dataset-specific validation rather than a
   generic cleanup.

2. `velocityDemo.py` imports heavy optional dependencies at package import time.
   Because `ADFWI/utils/__init__.py` imports model helpers directly,
   `import ADFWI.utils` also imports dependencies such as `obspy`, `h5py`, and
   `segyio`. This works in the current environment, but can make lightweight
   installs fragile.

3. Model download helpers rely on shell `wget`.
   The helpers do not expose checksum validation, retry policy, or a pure Python
   download path. They are acceptable for legacy examples, but should not be
   treated as a robust data-management layer.

4. Some dataset helpers use historical coordinate/key conventions.
   Different helpers return combinations of `x/y`, `x/z`, `dx/dy`, and `dx/dz`.
   This is compatible with existing examples but should be documented before
   any future API tightening.

5. `scipy.interpolate.interp2d` remains a future compatibility risk.
   `resample_marmousi_model` still uses `interp2d`. It is available in the
   current SciPy `1.13.1` environment, but this should be revisited if the
   dependency stack is upgraded.

6. Legacy mute helper names and arguments are intentionally preserved.
   `first_arrivel_picking.py`, `brutal_picker`, and the unused `mutetype`
   argument are historical API details. They should not be renamed casually
   unless all direct imports and examples are migrated.

7. Metric helpers preserve historical formulas.
   `MSE` and `RMSE` currently return sums/square-rooted sums rather than
   normalized mean errors. Tests lock this behavior, so changing naming or
   normalization would be a deliberate API break.

## Stop Decision

No clear bug or uncovered immediate contract gap was found that justifies
editing implementation code in this closeout round.

The utils stage should stop here unless one of the remaining risks is selected
as a bounded task with direct validation data.

## Recommended Next Bounded Task

If utils work continues, choose only one of these:

- add a lightweight import contract that documents which optional dependencies
  are required for `ADFWI.utils`;
- audit `velocityDemo.py` with dataset-free tests for synthetic builders only;
- defer utils entirely and move to the next framework module.
