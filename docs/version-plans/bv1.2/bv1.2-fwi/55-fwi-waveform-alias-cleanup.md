# 55 - FWI Waveform Alias Cleanup

## Goal

Make the acoustic and elastic FWI batch loops expose only waveform aliases that are consumed by the inversion path.

## Change

- In AcousticFWI.forward and AcousticFWI.forward_closure, keep only the pressure receiver waveform and pressure forward wavefield aliases used for loss construction and gradient processing.
- In ElasticFWI.forward, keep only the forward wavefield aliases used to accumulate pressure, vx, and vz gradient wavefields.
- Preserve the full record_waveform dictionary so elastic_synthetic_components can continue to construct configured synthetic loss components from the original propagator output.

## Scientific Contract

- Propagator calls, shot selection, observed-data indexing, transform preparation, misfit evaluation, regularization, backward propagation, gradient processing, and epoch updates are unchanged.
- Acoustic pressure inversion still uses record_waveform pressure data and forward_wavefield_p.
- Elastic component losses still come from elastic_synthetic_components(record_waveform), so receiver component semantics are unchanged.
- This cleanup should not change tensor values, gradient flow, or CPU/NPU comparison metrics.

## Validation

Completed validation:

- python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py: passed.
- conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py: 75 tests passed.
- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, vp_grad_norm 0.0, vp_update_norm 0.0.
  - Elastic CPU/NPU relative differences: loss 4.880620563312549e-06, vp_grad_norm 1.2590211728446073e-06, vp_update_norm 0.0.

## Next Optimization Direction

Continue reducing duplicated FWI loop scaffolding by extracting a small forward-record preparation helper that returns the shot_index, record_waveform, and gradient wavefield inputs while leaving acoustic and elastic loss semantics explicit in each driver.
