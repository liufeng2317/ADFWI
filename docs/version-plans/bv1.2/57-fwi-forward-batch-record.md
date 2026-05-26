# 57 - FWI Forward Batch Record

## Goal

Make the per-batch forward stage explicit in acoustic and elastic FWI loops without moving loss construction, regularization, gradient processing, or optimizer behavior out of the drivers.

## Change

- Added ForwardBatchRecord to carry the shot_index and propagator record_waveform for one batch.
- Added acoustic_forward_batch(propagator, batch_range, checkpoint_segments) for acoustic propagator execution.
- Added elastic_forward_batch(propagator, batch_range, fd_order=..., checkpoint_segments=...) for elastic propagator execution.
- Updated AcousticFWI.forward and AcousticFWI.forward_closure to use the forward batch record for observed-data indexing and pressure wavefield selection.
- Updated ElasticFWI.forward to use the forward batch record for elastic synthetic component construction, observed-data indexing, and gradient wavefield selection.
- Added focused tests that verify shot_index identity is preserved and elastic fd_order/checkpoint arguments are passed through unchanged.

## Scientific Contract

- Batch ranges still determine shot_index exactly as before.
- Acoustic propagator calls still receive the same shot_index and checkpoint_segments values.
- Elastic propagator calls still receive the same fd_order, shot_index, and checkpoint_segments values.
- Observed waveform indexing continues to use the same shot_index object paired with the forward record.
- Loss construction, regularization, backward propagation, gradient processing, optimizer stepping, scheduler stepping, and model constraints are unchanged.

## Validation

Completed validation:

- python -m py_compile ADFWI/fwi/runtime/forward.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py: passed.
- conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_backend_integration.py tests/test_fwi_data_contract.py: 79 tests passed.
- conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0: passed.
  - Acoustic CPU/NPU relative differences: loss 0.0, vp_grad_norm 0.0, vp_update_norm 0.0.
  - Elastic CPU/NPU relative differences: loss 4.880620563312549e-06, vp_grad_norm 1.2590211728446073e-06, vp_update_norm 0.0.

## Next Optimization Direction

Continue separating the FWI loop into explicit stages by extracting small loss-input preparation helpers for acoustic pressure and elastic components, while keeping transform application and component weighting visible enough for geophysical review.
