# TraceNormalize Shared Formula

## Goal

Make the transform-pipeline normalization path and the legacy FWI fallback
normalization path use one numerical definition. This prevents subtle drift
between `TraceNormalize()` and `ADFWI.fwi.data.normalize_waveform(...)` as the
framework continues to centralize waveform preprocessing.

## Current Step

A new low-level helper module, `ADFWI.fwi.normalization`, now owns the shared
`normalize_waveform(data, dim=1)` formula.

The public behavior is preserved:

1. `ADFWI.fwi.data.normalize_waveform` remains available by importing the shared
   helper.
2. `TraceNormalize(dim=...)` now calls the same helper and passes its configured
   dimension.
3. The formula still divides each trace by its maximum absolute amplitude and
   keeps all-zero traces finite by replacing zero denominators with 1.

The helper lives outside `data.py` and `transforms/amplitude.py` to avoid a
circular import: `data.py` imports transform classes while `TraceNormalize` now
needs the same normalization formula.

## Validation

The refactor was checked at transform, FWI-contract, and end-to-end smoke levels:

- `python -m py_compile ADFWI/fwi/normalization.py ADFWI/fwi/data.py ADFWI/fwi/transforms/amplitude.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_data_transforms.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_mute_transform_comparison.py tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 68 tests;
- transform tests now verify that `TraceNormalize()` equals
  `normalize_waveform(...)` for the default time dimension and for a custom
  dimension;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Update earlier transform-pipeline docs that still describe normalization as
   separate legacy and transform implementations.
2. Consider whether other small mathematical utilities should move into focused
   low-level helper modules before larger FWI-engine refactors.
3. Keep full CPU/NPU smoke comparisons after each data-path cleanup because FWI
   precision is more important than cosmetic simplification.
