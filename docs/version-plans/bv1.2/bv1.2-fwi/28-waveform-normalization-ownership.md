# Waveform Normalization Ownership

## Goal

Continue the bv1.2 module-organization pass by moving the canonical waveform-amplitude normalization formula next to the transform that owns the operation. This keeps `ADFWI.fwi` from accumulating flat helper modules while preserving old imports.

## Change

- Moved the canonical `normalize_waveform(data, dim=1)` implementation into `ADFWI.fwi.transforms.amplitude`, next to `TraceNormalize`.
- Exported `normalize_waveform` from `ADFWI.fwi.transforms` for the new transform-layer entry point.
- Kept `ADFWI.fwi.normalization.normalize_waveform` as a backward-compatible re-export.
- Kept `ADFWI.fwi.data.normalize_waveform` available through the data package re-export.

## Numerical Contract

The normalization formula is unchanged: each trace is divided by the maximum absolute amplitude along the selected dimension, and all-zero traces use denominator 1 to avoid NaN/Inf. This pass only changes ownership/import direction.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/transforms/amplitude.py ADFWI/fwi/transforms/__init__.py ADFWI/fwi/normalization.py ADFWI/fwi/data/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_data_transforms.py tests/test_fwi_data_contract.py` passed.
- `conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py` passed: 70 tests.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed.

CPU/NPU smoke comparison:

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 0.0 | pass |
| acoustic | vp_grad_norm | 0.0 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.880620563312549e-06 | pass |
| elastic | vp_grad_norm | 1.2590211728446073e-06 | pass |
| elastic | vp_update_norm | 0.0 | pass |

The smoke results match the previous bv1.2 baseline range, confirming the ownership/import pass does not change FWI numerical behavior.
