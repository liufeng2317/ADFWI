# FWI Misfit Evaluation Helper

## Goal

Move the duplicated AcousticFWI/ElasticFWI misfit-dispatch logic into the FWI
data contract layer while preserving each class's historical calling convention.
This keeps `calculate_loss()` focused on data preparation, optional
normalization, and one shared misfit evaluation call.

## Current Step

`ADFWI.fwi.data` now provides `evaluate_misfit_loss(...)`. AcousticFWI and
ElasticFWI call it from `calculate_loss()`.

The helper preserves existing behavior:

1. `Misfit` instances call `forward(synthetic, observed)`.
2. `Misfit_NIM` keeps its custom autograd signature with `p`, `trans_type`, and
   `theta`.
3. AcousticFWI keeps the legacy `.apply(...)` fallback for custom autograd
   Function-style losses.
4. ElasticFWI keeps the legacy callable fallback for losses invoked as
   `loss_fn(synthetic, observed)`.

## Validation

The helper dispatch rules and end-to-end smoke baseline were checked:

- added data-contract tests for `Misfit.forward`, callable fallback, and
  `.apply(...)` fallback;
- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 47 tests;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Keep DIP FWI untouched for now; it still uses older embedded logic and can be
   migrated separately after the core FWI path stabilizes.
2. Consider a later misfit API cleanup only after documenting which misfits are
   class instances, callables, or custom autograd Functions.
3. Continue validating with trace-missing smoke because it exercises both data
   preparation and loss evaluation in one path.
