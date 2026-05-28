# 54 - FWI Batch Range Cleanup

## Goal

Remove stale batch-bound local variables from the acoustic and elastic FWI iteration loops so each loop exposes only the values that drive the inversion step.

## Change

- Removed unused begin_index and end_index assignments from AcousticFWI.forward.
- Removed unused begin_index and end_index assignments from AcousticFWI.forward_closure.
- Removed unused begin_index and end_index assignments from ElasticFWI.forward.
- Kept batch_range itself intact because it still carries shot_index for forward modeling and begin/end metadata for single-batch progress descriptions.

## Scientific Contract

- Forward modeling still receives the same shot_index tensors from iter_batch_ranges.
- Loss construction, regularization, backward propagation, gradient post-processing, optimizer stepping, scheduler stepping, and model constraint application are unchanged.
- This is a readability and maintainability cleanup; it should not change acoustic or elastic inversion numerics.

## Validation

Completed validation:

- python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py: passed.
- conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py: 75 tests passed.
- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, vp_grad_norm 0.0, vp_update_norm 0.0.
  - Elastic CPU/NPU relative differences: loss 4.880620563312549e-06, vp_grad_norm 1.2590211728446073e-06, vp_update_norm 0.0.

## Next Optimization Direction

Audit unused waveform unpacking in the acoustic and elastic loops. The likely next step is to keep full record_waveform dictionaries for loss-component construction while removing local receiver or wavefield aliases that are never consumed by the current inversion path.
