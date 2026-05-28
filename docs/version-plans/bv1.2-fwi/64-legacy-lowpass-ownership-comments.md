# 64 - Legacy Low-Pass Ownership Comments

## Optimization Path

Clarify the multiscale low-pass compatibility boundary without changing
implementation behavior.

The codebase currently has two user-visible paths:

- `ADFWI.fwi.multiscale.legacy_lowpass`: implementation owner for the historical
  SciPy Butterworth/filtfilt low-pass path and custom autograd wrapper;
- `ADFWI.fwi.multiScaleProcessing`: backward-compatible re-export module for
  older scripts.

This step only updates comments and docstrings so future optimization work does
not confuse the legacy Butterworth path with the newer pure torch
`LowPassFilter`.

## Change

- Rewrote the module docstring in `ADFWI.fwi.multiscale.legacy_lowpass` to state
  implementation ownership, CPU NumPy/SciPy round-trip behavior, and why it
  remains separate from the pure torch FIR low-pass transform.
- Removed the long commented-out historical frequency-domain draft from
  `legacy_lowpass.py`.
- Added concise docstrings for `lowpass`, `adj_lowpass`, `data2d_to_3d`,
  `data3d_to_2d`, `lpass`, and `Lfilter`.
- Updated `ADFWI.fwi.multiScaleProcessing` to clarify that it intentionally
  contains only compatibility re-exports.

## Scientific Contract

- No executable filtering logic changed.
- `lowpass`, `adj_lowpass`, `data2d_to_3d`, `data3d_to_2d`, `lpass`, and
  `Lfilter` keep the same call signatures.
- `ADFWI.fwi.multiScaleProcessing` continues to re-export the same objects.
- `LegacyLowPassFilter` continues to route through the legacy `lpass` path.
- The CPU NumPy/SciPy boundary and custom backward path remain unchanged.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/multiscale/legacy_lowpass.py ADFWI/fwi/multiScaleProcessing.py tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py`: 7 tests passed.
- `conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_mute_transform_comparison.py tests/test_fwi_data_contract.py`: 49 tests passed.

Because this step only changes comments/docstrings and removes dead commented
code, no FWI core numerical smoke was required. The low-pass comparison tests
still confirm exact `LegacyLowPassFilter`/`lpass` agreement and finite backward
gradients.

## Next Optimization Direction

Add a short user-facing multiscale/low-pass note to the script example guide or
FWI module map. It should tell users when to choose legacy-compatible low-pass
filtering versus the differentiable torch `LowPassFilter`, and should point to
the existing comparison tests and bv1.2 low-pass decision record.
