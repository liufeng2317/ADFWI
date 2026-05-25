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

At this step, `calculate_loss(..., apply_transforms=True)` was intentionally left
unchanged. That follow-up has since been completed in
`18-calculate-loss-receiver-selection.md`: direct `calculate_loss()` calls now
reuse `_prepare_loss_pair()` when transforms are enabled. The training loops
still call `_prepare_loss_pair()` first and then use
`calculate_loss(..., apply_transforms=False)` to avoid double preprocessing.

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

1. The public `calculate_loss(..., apply_transforms=True)` receiver-selection
   follow-up is complete; keep the direct trace-missing tests as API guards.
2. Continue keeping propagator kernels untouched until FWI orchestration
   helpers are stable.
3. Use later docs for current next steps: normalization sharing is tracked in
   `21-trace-normalize-shared-formula.md`, and broader engine cleanup remains a
   separate staged task.
