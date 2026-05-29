# 02 - View Explicit Exports

## Goal

Replace the wildcard import in `ADFWI/view/__init__.py` with explicit plotting
helper exports while preserving the intended `ADFWI.view` plotting API.

## Scope

- `ADFWI/view/__init__.py`
- `tests/test_view_contracts.py`
- `docs/version-plans/bv1.2-view/`

No plotting implementation or Matplotlib backend behavior is changed.

## What Changed

- Added a package docstring that defines `ADFWI.view` as a plotting helper
  namespace.
- Replaced `from .inverted_loss_model import *` with explicit imports:
  - `plot_misfit`
  - `plot_initial_and_inverted`
  - `animate_inversion_process`
- Added `__all__` for the public plotting helper names.
- Extended `tests/test_view_contracts.py` to assert the explicit public plotting
  surface.

The previous wildcard import also exposed implementation details such as
`np`, `plt`, `os`, `animation`, and `HTML` at package level. These are not
treated as intended plotting API names and are not included in `__all__`.

## Public Plotting Surface

- `animate_inversion_process`
- `plot_bcx_bcz`
- `plot_damp`
- `plot_eps_delta_gamma`
- `plot_initial_and_inverted`
- `plot_lam_mu`
- `plot_misfit`
- `plot_model`
- `plot_survey`
- `plot_vp_rho`
- `plot_vp_vs_rho`
- `plot_waveform2D`
- `plot_waveform_trace`
- `plot_waveform_wiggle`
- `plot_wavelet`

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_view_contracts.py
conda run -n adfwi python -m py_compile ADFWI/view/*.py tests/test_view_contracts.py
conda run -n adfwi python -c "from ADFWI.view import plot_misfit, plot_initial_and_inverted, animate_inversion_process, plot_waveform2D, plot_damp; import ADFWI.view as view; print(sorted(view.__all__))"
git diff --check
```

## Result

- View contract tests passed: 6 tests.
- `py_compile` passed.
- Explicit import smoke passed.
- Only existing local NPU/Ascend and Matplotlib/PyParsing warnings were printed.

## Next Bounded Task

Run a short view closeout audit. If no clear plotting bug remains, stop the
view stage rather than continuing readability cleanup.
