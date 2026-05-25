# FWI Loss Pair Builder

## Goal

Move the shared AcousticFWI/ElasticFWI pre-loss pair preparation into the FWI
data contract layer. This keeps receiver selection, transform context assembly,
and data transform pipeline execution in one reusable path while preserving the
historical order before misfit evaluation.

## Current Step

`ADFWI.fwi.data` now provides `prepare_fwi_loss_pair(...)`. AcousticFWI and
ElasticFWI use this helper inside their `_prepare_loss_pair()` wrappers.

The helper preserves the established ordering:

1. Build the same transform context as `build_fwi_transform_context(...)`.
2. Select or mask receivers when `shot_index` provides a receiver mask.
3. Run the configured data transform pipeline on same-shape synthetic/observed
   tensors.

`calculate_loss(..., apply_transforms=True)` was intentionally left unchanged in
this step. The training loops already call `_prepare_loss_pair()` and then call
`calculate_loss(..., apply_transforms=False)`, so this refactor avoids changing
public API behavior while cleaning the main inversion path.

## Validation

The helper was checked with contract tests and end-to-end CPU/NPU smoke runs:

- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 41 tests;
- the first smoke run caught a missing `prepare_fwi_loss_pair` import in
  ElasticFWI, which was fixed before final validation;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed after the import fix.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Decide whether public `calculate_loss(..., apply_transforms=True)` should
   also use receiver selection when `shot_index` is provided.
2. If that API behavior is changed, add explicit tests for trace-missing
   receiver dimensions before merging.
3. Continue keeping propagator kernels untouched until the FWI orchestration
   layer is stable.
