# FWI Loop Structure

## Goal

The next cleanup layer is the acoustic/elastic FWI main loop. The intent is to
reduce duplicated orchestration code while keeping numerical work in the
existing model-specific paths until each extraction is tested.

## Current Step

Implemented a shared `iter_batch_ranges` helper in `ADFWI.fwi.loop` and routed
AcousticFWI and ElasticFWI batch iteration through it. This helper only owns shot
range generation:

- `batch_size=None` keeps full-batch behavior.
- `batch_size > n_shots` keeps full-batch behavior.
- partial final batches keep the same contiguous shot ranges as the previous
  `math.ceil(n_shots / batch_size)` logic.
- invalid `n_shots` or `batch_size` values fail early with `ValueError`.

No propagation, transform, misfit, regularization, backward, gradient
processing, optimizer, or scheduler behavior was moved in this step.

## Validation

Unit tests cover full-batch, oversized batch, partial final batch, and invalid
input behavior in `tests/test_fwi_loop.py`.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/loop.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_loop.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_loop.py tests/test_backend_integration.py tests/test_fwi_data_contract.py` passed: 30 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed. Acoustic CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`; elastic maximum relative drift was `4.880620563312549e-06`, within the `1e-5` tolerance.

Smoke comparison remained stable because the helper only constructs the same
`shot_index` arrays passed to the propagators.

## Next Steps

1. Extract per-batch loss accumulation only after acoustic and elastic smoke
   tests remain stable.
2. Consider a small per-epoch state object for cached forward wavefields and
   scalar loss history.
3. Keep LBFGS/NLCG closure behavior separate until normal optimizer paths are
   fully covered.
