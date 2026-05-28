# 116. FWI Data Context Helper

## Goal

Start the post-gradient-processor optimization phase with a small
data-contract cleanup in `ADFWI.fwi.data.preparation`. The change tightens the
responsibility boundary inside transform-context construction without changing
public API or numerical behavior.

## Optimization Path

1. Add a private `_shot_scoped_context_values(...)` helper in
   `ADFWI/fwi/data/preparation.py`.
2. Move receiver mask, source x, receiver x, and data-mask shot selection into
   that helper.
3. Keep `build_fwi_transform_context(...)` as the public owner for FWI transform
   context construction.
4. Add a focused test that locks the existing `propagator_dt` precedence over
   `default_dt`.

## Numerical Contract

No FWI core numerical formula changed. The helper preserves existing behavior:

- shot-scoped values are omitted when `shot_index is None`;
- `receiver_masks_2d`, `src_x`, and `data_masks` are indexed by `shot_index`;
- `rcv_x` is moved to CPU as before;
- `propagator_dt` still takes precedence over `default_dt`.

## Validation Result

```bash
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_data_public_api.py
# Ran 30 tests in 0.039s, OK

conda run -n adfwi python -m py_compile ADFWI/fwi/data/preparation.py tests/test_fwi_data_contract.py
# OK

conda run -n adfwi python -m unittest tests/test_import_surface_policy.py
# Ran 2 tests in 2.962s, OK

conda run -n adfwi python -m unittest tests/test_fwi_runtime.py
# Ran 24 tests in 0.227s, OK
```

## Next Direction

Continue small FWI data-contract cleanups only when they reduce local
responsibility mixing or strengthen tests. Avoid changing loss-input shapes,
receiver selection order, or transform order without a full numerical
comparison.
