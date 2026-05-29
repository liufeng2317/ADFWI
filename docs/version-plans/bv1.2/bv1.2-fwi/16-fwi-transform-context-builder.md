# FWI Transform Context Builder

## Goal

Move the shared transform-context assembly from AcousticFWI and ElasticFWI into
the FWI data contract layer. This keeps the FWI classes focused on inversion
orchestration while preserving the exact context fields passed to data
transforms before loss evaluation.

## Current Step

`ADFWI.fwi.data` now provides `build_fwi_transform_context(...)`. AcousticFWI
and ElasticFWI use it inside their `_build_transform_context()` wrappers.

The helper preserves the previous behavior exactly:

- `shot_index`, `cutoff_freq`, `dt`, `late_window`, `offset_mute_threshold`, and
  `dx` are always present in the context;
- receiver mask, source x locations, receiver x locations, and data mask are
  included only when `shot_index` is provided;
- `propagator_dt` overrides the default propagator `dt`; otherwise default `dt`
  is used;
- source and receiver x tensors are moved to CPU before being placed in the
  context, matching the previous AcousticFWI/ElasticFWI logic.

## Validation

The helper was checked at both contract and end-to-end smoke levels:

- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 40 tests;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the existing bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic max relative drift was `4.880620563312549e-06`, below the `1e-5`
  tolerance.

## Next Steps

1. Consider unifying `_prepare_loss_pair()` wrappers once transform context
   construction is stable.
2. Keep receiver selection order locked by data contract tests before changing
   loss preparation APIs.
3. Continue avoiding propagator kernel refactors until FWI orchestration helpers
   are stable.
