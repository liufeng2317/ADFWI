# Elastic Component Loss Inputs

## Goal

Move elastic component selection and component-weight lookup out of the
`ElasticFWI.forward()` loop. This keeps the inversion loop focused on propagation,
loss evaluation, regularization, and optimization while leaving the elastic
component contract in `ADFWI.fwi.data`.

## Current Step

`ADFWI.fwi.data` now provides `elastic_component_loss_inputs(...)`. The helper
returns active component loss inputs in the stable `ELASTIC_COMPONENTS` order:

```text
(component_name, synthetic_component, observed_component, component_weight)
```

`ElasticFWI.forward()` now uses this helper before calling `_prepare_loss_pair()`
and `calculate_loss(...)`. Loss evaluation remains inside `ElasticFWI`, so this
step does not change transform application, misfit dispatch, autograd behavior,
or component loss accumulation semantics.

The helper intentionally preserves the current ordering behavior: user-provided
`inversion_component` order does not control evaluation order; components are
evaluated in `("pressure", "vx", "vz")` order when active.

## Validation

The refactor was checked at data-contract and end-to-end smoke levels:

- `python -m py_compile ADFWI/fwi/data.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py`
  passed;
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py`
  passed with 49 tests;
- data-contract tests verify active component filtering, stable component order,
  tensor identity preservation, weight lookup, and empty active component lists;
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0`
  passed.

CPU/NPU smoke drift stayed within the established bv1.2 baseline:

- acoustic loss, gradient norm, and update norm drift were all zero;
- elastic loss relative drift was `4.880620563312549e-06`;
- elastic gradient relative drift was `1.2590211728446073e-06`;
- elastic update norm drift was zero.

## Next Steps

1. Consider a dedicated helper for summing weighted component losses after
   verifying multi-component elastic smoke behavior.
2. Add or extend a multi-component elastic smoke case such as
   `pressure,vx,vz` before deeper FWI engine extraction.
3. Keep component selection in `ADFWI.fwi.data`; keep actual misfit evaluation in
   FWI classes until the shared engine boundary is clearer.
