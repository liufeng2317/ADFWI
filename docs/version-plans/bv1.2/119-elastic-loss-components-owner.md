# Elastic Loss Components Owner

## Goal

Move the supported elastic loss component list out of `ADFWI.fwi.iteration.loss`.

`loss.py` should construct and evaluate batch losses, but it should not define
the global set of elastic FWI components. That ownership belongs to the elastic
FWI driver because the supported components are part of the elastic inversion
configuration.

## Change

Added the driver-owned constant in `ADFWI/fwi/elastic_fwi.py`:

```python
ELASTIC_LOSS_COMPONENTS = ("pressure", "vx", "vz")
```

Updated `ADFWI.fwi.iteration.loss` so helper functions receive the supported
component order explicitly:

- `normalize_elastic_component_weights(..., supported_components=...)`
- `elastic_component_loss_inputs(..., component_order)`
- `elastic_loss_inputs(..., component_order, shot_index)`

Updated `ADFWI.fwi.iteration.step.apply_elastic_batch_loss_step(...)` to accept
`elastic_loss_components` and pass it into loss-input construction.

This avoids any reverse import from `iteration.loss` into `elastic_fwi.py`, so
there is no circular import.

## Numerical Scope

This is an ownership and dependency-direction change only. It does not change:

- supported component values;
- elastic component order;
- pressure construction;
- component weighting;
- transform order;
- misfit dispatch;
- backward/update order.

## Validation

```bash
python -m py_compile \
  ADFWI/fwi/iteration/loss.py \
  ADFWI/fwi/iteration/step.py \
  ADFWI/fwi/elastic_fwi.py \
  tests/test_fwi_iteration_loss.py \
  tests/test_fwi_iteration.py

conda run -n adfwi python -m unittest \
  tests/test_fwi_iteration_loss.py \
  tests/test_fwi_iteration.py \
  tests/test_fwi_runtime.py \
  tests/test_import_surface_policy.py

conda run -n adfwi python -m unittest \
  tests/test_backend_integration.py \
  tests/test_data_transforms.py

git diff --check
```

All checks passed.

## Next Direction

Keep elastic-FWI configuration constants in the elastic driver unless another
module genuinely owns the same physical contract across multiple workflows.
