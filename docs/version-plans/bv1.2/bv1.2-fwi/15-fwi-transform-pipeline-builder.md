# FWI Transform Pipeline Builder

## Goal

Move the shared default data-transform pipeline construction out of AcousticFWI
and ElasticFWI into the FWI data contract layer. This keeps the FWI classes
focused on inversion orchestration while preserving the existing pre-loss
transform order and public behavior.

## Current Step

`ADFWI.fwi.data` now provides `build_fwi_data_transform_pipeline(
data_transform_pipeline, waveform_normalize)`. AcousticFWI and ElasticFWI use
this helper inside their `_configure_data_transform_pipeline()` wrappers.

The helper preserves the previous order exactly:

1. `LegacyOffsetMute(required=False)`
2. `LegacyLateWindowMute(required=False)`
3. `LegacyLowPassFilter(required=False)`
4. `DataMask(required=False, apply_to="synthetic")`
5. `TraceNormalize()` only when no custom pipeline is supplied and
   `waveform_normalize=True`
6. custom user pipeline appended after the legacy-compatible transforms when
   supplied

When a custom pipeline is supplied, `waveform_normalize` is returned unchanged,
matching the previous AcousticFWI/ElasticFWI behavior.

## Validation

Validation covers pipeline ordering and end-to-end numerical behavior:

- unit tests verify default transform order and custom pipeline append behavior;
- syntax compilation includes data, acoustic, and elastic FWI modules;
- existing backend/data/iteration tests remain green;
- acoustic and elastic CPU/NPU smoke comparisons remain within the established
  tolerances.

Validation run after the change:

- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py` passed: 38 tests OK.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed. Acoustic CPU/NPU drift was zero for loss, `vp_grad_norm`, and `vp_update_norm`; elastic maximum relative drift was `4.880620563312549e-06`, within the `1e-5` tolerance.

## Next Steps

1. Consider moving `_build_transform_context()` into a shared helper next, since
   acoustic and elastic still build the same context structure.
2. Keep custom pipeline behavior locked by tests before changing user-facing
   transform APIs.
3. Avoid modifying low-pass or mute implementations unless numerical comparison
   tests are expanded for those transforms.
