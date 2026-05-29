# 103. Data Facade User Documentation

## Goal

Document the public import policy created in record 102:

- `ADFWI.fwi.data` is a stable user-facing data-contract facade.
- `ADFWI.fwi.iteration` and `ADFWI.fwi.runtime` are namespace packages.
- Framework internals should keep using owner modules.

## Optimization Path

1. Add a `FWI Data Contract API` section to `docs/backend-usage.md`.
2. Show the recommended user import style:

   ```python
   from ADFWI.fwi.data import (
       build_fwi_data_transform_pipeline,
       prepare_fwi_loss_pair,
       normalize_elastic_component_weights,
   )
   ```

3. Explicitly state that internal FWI code should use owner modules such as
   `ADFWI.fwi.data.preparation`, `ADFWI.fwi.data.loss`, and
   `ADFWI.fwi.data.components`.
4. Clarify that `ADFWI.fwi.iteration` and `ADFWI.fwi.runtime` are namespace
   packages in bv1.2 and should not be used as broad import surfaces.

## Numerical Contract

Documentation-only change. No code path, tensor operation, FWI helper, or test
fixture changed.

## Validation Result

Focused public API import smoke confirms the documented imports work:

```bash
conda run -n adfwi python -c "from ADFWI.fwi.data import build_fwi_data_transform_pipeline, prepare_fwi_loss_pair, normalize_elastic_component_weights; print(build_fwi_data_transform_pipeline.__name__, prepare_fwi_loss_pair.__name__, normalize_elastic_component_weights.__name__)"
# build_fwi_data_transform_pipeline prepare_fwi_loss_pair normalize_elastic_component_weights
```

The public API test that locks the facade passed:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_data_public_api.py
# Ran 2 tests in 0.000s, OK
```

## Next Direction

Continue with real optimization work after the public/internal import policy is
settled. The next likely code-facing cleanup is to audit examples and notebooks
for old import styles or stale comments, without changing numerical behavior.
