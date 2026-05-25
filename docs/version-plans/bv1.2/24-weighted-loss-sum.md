# Weighted Loss Sum Helper

## Goal

Move elastic component loss summation into a small shared helper so the FWI loop only expresses component selection, pair preparation, misfit evaluation, and final data-loss assembly.

## Change

- Added `sum_weighted_losses` in `ADFWI.fwi.data`.
- Replaced ElasticFWI hand-written component loss accumulation with the shared helper.
- Kept the non-empty path initialized from the first real loss tensor so device, dtype, and autograd history are preserved.
- Kept an explicit empty-component fallback returning scalar zero on the requested device.

## Numerical Contract

This is an algebra-only refactor. For non-empty component losses, the helper uses the same ordered pairwise addition as the previous ElasticFWI implementation. It should therefore preserve data loss values and gradients exactly for the same inputs.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py` passed: 51 tests.
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

The elastic CPU/NPU drift remains within the existing `1e-05` smoke tolerance and matches the previous bv1.2 baseline range.
