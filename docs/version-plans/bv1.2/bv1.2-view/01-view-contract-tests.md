# 01 - View Contract Tests

## Goal

Add focused smoke coverage for the public `ADFWI/view` plotting helpers and fix
only a confirmed plotting bug exposed by that coverage.

## Scope

- `tests/test_view_contracts.py`
- `ADFWI/view/velocity_model.py`
- `docs/version-plans/bv1.2-view/`

No plotting style, layout policy, backend policy, or numerical behavior is
changed.

## What Changed

1. Added `tests/test_view_contracts.py`.
   The tests use tiny synthetic arrays and check that public plotting helpers
   can save figures and close figures when `show=False`.

2. The test file selects a non-interactive Matplotlib backend locally for
   headless execution before importing `ADFWI.view`.
   The `ADFWI/view` library modules themselves do not set or change the
   Matplotlib backend.

3. Fixed `plot_eps_delta_gamma()` in `velocity_model.py`.
   In the branch where `dx <= 0` or `dz <= 0`, `delta` was drawn on `ax[0]`
   instead of `ax[1]`. The contract test observed image counts `[2, 0, 1]`
   across the three panels. After the fix, each panel owns one image:
   `[1, 1, 1]`.

## Covered Helpers

- `plot_vp_rho`
- `plot_vp_vs_rho`
- `plot_eps_delta_gamma`
- `plot_lam_mu`
- `plot_model`
- `plot_bcx_bcz`
- `plot_damp`
- `plot_survey`
- `plot_wavelet`
- `plot_waveform_trace`
- `plot_waveform2D`
- `plot_waveform_wiggle`
- `plot_misfit`
- `plot_initial_and_inverted`
- `animate_inversion_process`

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_view_contracts.py
conda run -n adfwi python -m py_compile ADFWI/view/*.py tests/test_view_contracts.py
git diff --check
```

## Result

- View contract tests passed: 5 tests.
- `py_compile` passed for `ADFWI/view/*.py` and
  `tests/test_view_contracts.py`.
- `git diff --check` passed.
- Only existing local NPU/Ascend and Matplotlib/PyParsing warnings were printed.

## Next Bounded Task

Make `ADFWI/view/__init__.py` explicit:

- add a package docstring;
- replace wildcard import from `inverted_loss_model.py` with explicit imports;
- preserve the current public names;
- validate with `tests/test_view_contracts.py`, import smoke, and `py_compile`.
