# 56 - FWI Wavefield Input Helper

## Goal

Move small record-to-wavefield selection rules out of the acoustic and elastic FWI loops while keeping loss construction and model-specific inversion semantics explicit in each driver.

## Change

- Added acoustic_pressure_waveforms(record_waveform) for selecting acoustic pressure receiver data and pressure forward wavefield input.
- Added elastic_gradient_wavefields(record_waveform, inversion_components) for selecting the elastic wavefields accumulated for gradient processing.
- Updated AcousticFWI.forward and AcousticFWI.forward_closure to use the acoustic helper.
- Updated ElasticFWI.forward to use the elastic helper while leaving elastic_synthetic_components(record_waveform) as the source of configured loss components.
- Added focused runtime tests for acoustic pressure selection and elastic pressure sign convention.

## Scientific Contract

- Propagator calls, shot indexing, observed-data indexing, transform preparation, misfit evaluation, regularization, backward propagation, optimizer stepping, scheduler stepping, and model constraints are unchanged.
- Acoustic pressure inversion still uses record_waveform["p"] for the synthetic waveform and record_waveform["forward_wavefield_p"] for legacy gradient processing.
- Elastic pressure gradient wavefield keeps the legacy sign convention: pressure = -(forward_wavefield_txx + forward_wavefield_tzz).
- Elastic loss components still come from elastic_synthetic_components(record_waveform), so component weighting and receiver semantics are unchanged.

## Validation

Completed validation:

- python -m py_compile ADFWI/fwi/runtime/wavefield.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py: passed.
- conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_backend_integration.py tests/test_fwi_data_contract.py: 77 tests passed.
- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, vp_grad_norm 0.0, vp_update_norm 0.0.
  - Elastic CPU/NPU relative differences: loss 4.880620563312549e-06, vp_grad_norm 1.2590211728446073e-06, vp_update_norm 0.0.

## Next Optimization Direction

Continue separating the FWI loop into explicit stages by extracting a small per-batch forward stage helper or dataclass that carries shot_index, record_waveform, synthetic loss inputs, and gradient wavefield inputs without hiding acoustic/elastic physics.
