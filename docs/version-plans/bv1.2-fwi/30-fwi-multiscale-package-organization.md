# FWI Multiscale Package Organization

## Goal

Continue the bv1.2 module-organization pass by moving the legacy multiscale low-pass implementation out of the camelCase root-level `multiScaleProcessing.py` module into a responsibility-focused package, without changing the legacy numerical path.

## Change

- Added `ADFWI.fwi.multiscale`.
- Moved the legacy SciPy Butterworth/filtfilt low-pass implementation to `ADFWI.fwi.multiscale.legacy_lowpass`.
- Kept `ADFWI.fwi.multiScaleProcessing` as a backward-compatible re-export module.
- Added compatibility tests confirming the old module re-exports the new package objects and that 2D/3D reshaping helpers preserve values.

## Numerical Contract

This is an import-organization refactor only. The legacy low-pass formula, SciPy `filtfilt` path, custom autograd wrapper, 2D/3D reshape helpers, and adjoint low-pass implementation are preserved. `LegacyLowPassFilter` and historical imports from `ADFWI.fwi.multiScaleProcessing` continue to use the same objects.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/multiscale/__init__.py ADFWI/fwi/multiscale/legacy_lowpass.py ADFWI/fwi/multiScaleProcessing.py ADFWI/fwi/transforms/filters.py tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py` passed.
- `conda run -n adfwi python -m unittest tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py tests/test_mute_transform_comparison.py tests/test_data_transforms.py` passed: 29 tests.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases legacy-lowpass --devices cpu,npu:0` passed.

CPU/NPU legacy-lowpass smoke comparison:

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 7.481579229699596e-08 | pass |
| acoustic | vp_grad_norm | 1.9492708119015068e-07 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.079844737428672e-07 | pass |
| elastic | vp_grad_norm | 1.665177581614308e-07 | pass |
| elastic | vp_update_norm | 1.0913013244469394e-07 | pass |

The legacy-lowpass smoke results remain within the established `1e-05` tolerance and match the previous bv1.2 low-pass baseline range.
