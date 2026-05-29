# FWI Waveform Normalization Helper

## Goal

Move the legacy fallback waveform normalization formula shared by AcousticFWI
and ElasticFWI into the FWI data contract layer. This keeps normalization
behavior in one place while preserving the existing `_normalize()` wrappers for
compatibility with existing notebooks and research scripts.

## Current Step

`ADFWI.fwi.data` now provides `normalize_waveform(data)`. AcousticFWI and
ElasticFWI call this helper from their `_normalize()` methods.

The helper preserves the historical behavior exactly:

1. Normalize each trace over time dimension 1.
2. Use the maximum absolute amplitude as denominator.
3. Replace all-zero trace denominators with 1 so zero traces remain zero.

This is the fallback normalization path used when `waveform_normalize` remains
true after pipeline configuration, for example with custom data transform
pipelines that intentionally leave legacy normalization outside the pipeline.
The default bv1.2 path still uses `TraceNormalize()` in the internal transform
pipeline and sets `waveform_normalize=False`.

## Validation

The helper was checked against the legacy formula and the main FWI smoke
baseline:

- added a data-contract test for nonzero traces and all-zero traces;
- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 44 tests;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Consider whether `TraceNormalize` should internally reuse `normalize_waveform`
   to make the transform and fallback paths share one formula.
2. If that is done, compare transform-only tests and full smoke results because
   transform semantics are user-facing.
3. Keep this helper small and formula-only; component selection and receiver
   handling should remain in the loss-pair helpers.
