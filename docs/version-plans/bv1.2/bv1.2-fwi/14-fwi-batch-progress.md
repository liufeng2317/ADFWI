# FWI Batch Progress

## Goal

Move non-numerical batch progress display logic from AcousticFWI and ElasticFWI
into the iteration helper module. This makes `ADFWI.fwi.iteration` responsible
for another small piece of iteration orchestration without touching propagation,
loss construction, backward propagation, gradient processing, optimizer steps,
or scheduler steps.

## Current Step

`ADFWI.fwi.iteration` now provides `set_batch_description(progress_bar,
batch_range, batch_count)`. It preserves the historical behavior exactly:

- when there is one batch, set the tqdm description to `Shot:{begin} to {end}`;
- when there are multiple batches, leave the progress description unchanged.

AcousticFWI normal/closure paths and ElasticFWI now call this helper instead of
checking `len(batch_ranges) == 1` directly.

## Validation

Validation covers the UI helper behavior and confirms the FWI numerical baseline
remains unchanged:

- unit tests verify single-batch and multi-batch description behavior;
- syntax compilation includes iteration, acoustic, and elastic FWI modules;
- existing backend/data/iteration tests remain green;
- acoustic and elastic CPU/NPU smoke comparisons remain within the established
  tolerances.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/iteration.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_backend_integration.py tests/test_fwi_data_contract.py` passed: 36 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed. Acoustic CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`; elastic maximum relative drift was `4.880620563312549e-06`, within the `1e-5` tolerance.

## Next Steps

1. Consider extracting a small epoch/batch progress-bar factory only after this
   description helper remains stable.
2. Keep propagation and gradient-related accumulators separate from UI helpers.
3. Continue using CPU/NPU smoke checks whenever changes are close to loss,
   backward, or gradient behavior.
