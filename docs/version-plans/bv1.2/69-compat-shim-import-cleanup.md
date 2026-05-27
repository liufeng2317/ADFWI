# 69 - Compatibility Shim Import Cleanup

## Optimization Path

Respond to the observation that ADFWI.fwi.normalization and
ADFWI.fwi.multiScaleProcessing are now thin files. They are intentionally kept as
backward-compatible import shims, but internal code should depend on the
canonical implementation owners instead of routing through those shims.

## Change

- Updated ADFWI.fwi.data to import normalize_waveform directly from
  ADFWI.fwi.transforms.amplitude.
- Updated LegacyLowPassFilter to import lpass directly from ADFWI.fwi.multiscale.
- Updated DIP acoustic/elastic FWI legacy low-pass imports to use
  ADFWI.fwi.multiscale and removed a duplicate lpass import in dip_elastic_fwi.
- Kept ADFWI.fwi.normalization and ADFWI.fwi.multiScaleProcessing in place as
  compatibility shims for old user scripts and compatibility tests.
- Extended data transform tests to assert that the legacy normalization import,
  transform package export, and canonical transform implementation are the same
  function object.

## Scientific Contract

- No waveform normalization formula changed.
- No legacy low-pass implementation changed.
- No FWI propagator, misfit, optimizer, transform order, or gradient behavior
  changed.
- This is an import-ownership cleanup only. Because no executable numerical
  formula changed, full FWI numerical smoke is not required for this step.
- Compatibility tests continue to exercise the old shim imports so accidental
  removal is caught before a planned breaking release.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_fwi_data_contract.py tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py: 51 tests passed.
- conda run -n adfwi python -m py_compile ADFWI/fwi/normalization.py ADFWI/fwi/multiScaleProcessing.py ADFWI/fwi/data/__init__.py ADFWI/fwi/transforms/filters.py ADFWI/dip/dip_acoustic_fwi.py ADFWI/dip/dip_elastic_fwi.py: passed.
- rg confirmed that remaining ADFWI.fwi.normalization and ADFWI.fwi.multiScaleProcessing imports are compatibility tests, not production FWI internals.

## Decision

Do not delete the thin files in bv1.2. They are cheap compatibility shims and
protect older scripts. A future breaking release can remove them after adding
explicit deprecation warnings and migration notes.

## Next Optimization Direction

Benchmark TorchGradProcessor inside acoustic mini-inversion and add a script
option for users to opt into torch-native gradient processing once loss,
gradient-norm, and model-update drift are documented.
