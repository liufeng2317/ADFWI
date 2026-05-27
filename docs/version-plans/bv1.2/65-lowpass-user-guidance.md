# 65 - Low-Pass User Guidance

## Optimization Path

Continue from step 64 by moving the clarified low-pass ownership boundary into
user-facing guidance.

The goal is to help researchers choose between:

- the legacy-compatible low-pass path used to reproduce existing FWI workflows;
- the pure torch `LowPassFilter`, which is differentiable but numerically
  different from the legacy Butterworth/filtfilt path.

## Change

- Added a "Choosing Low-Pass Filtering" section to `scripts/examples/README.md`.
- Documented the quick rule:
  - use `cutoff_freq` in FWI drivers or `LegacyLowPassFilter` for old workflow
    reproducibility;
  - use `LowPassFilter` for new differentiable transform experiments;
  - treat legacy-to-torch replacement as a numerical-method change.
- Pointed users to `docs/version-plans/bv1.2/04-lowpass-filter-comparison.md`
  and the focused low-pass tests.
- Updated the FWI module map to clarify that `ADFWI.fwi.multiscale` owns the
  legacy-compatible filtering path.

## Scientific Contract

- No Python implementation code changed.
- No low-pass formula, transform order, or FWI data path changed.
- Existing `cutoff_freq` behavior still routes through `LegacyLowPassFilter` in
  the default FWI transform pipeline.
- Pure torch `LowPassFilter` remains available as a separate transform, not a
  drop-in replacement for legacy FWI results.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/multiscale/legacy_lowpass.py ADFWI/fwi/multiScaleProcessing.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py`: 7 tests passed.

Because this step only changes documentation, no FWI core numerical smoke was
required. The low-pass comparison tests remain the numerical guard for legacy
low-pass exactness and finite backward behavior.

## Next Optimization Direction

Add an elastic component usage note or tiny script option example showing
pressure-only versus pressure/vx/vz inversion. This is a user-facing geophysical
choice and complements the existing component-weight helpers.
