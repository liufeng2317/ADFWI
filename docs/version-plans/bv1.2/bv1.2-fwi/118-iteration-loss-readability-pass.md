# Iteration Loss Readability Pass

## Goal

Make `ADFWI/fwi/iteration/loss.py` easier to read after the iteration helper
rebalance.

The previous structure was functionally correct but still carried two sources
of confusion:

- the file did not clearly show where records, component helpers, pair
  preparation, misfit dispatch, and batch-loss merge started and ended;
- `build_fwi_data_transform_pipeline(...)` kept the old `data` wording even
  though `ADFWI.fwi.data` has been removed.

## Change

1. Added explicit section markers inside `ADFWI/fwi/iteration/loss.py`:
   - loss records;
   - elastic component bookkeeping;
   - loss input construction;
   - loss pair preparation;
   - misfit dispatch and weighted loss evaluation;
   - batch loss merge.
2. Updated the module docstring to state that `loss.py` stops at producing loss
   tensors/scalars and that one-batch forward/backward execution lives in
   `ADFWI.fwi.iteration.step`.
3. Renamed:

```text
build_fwi_data_transform_pipeline(...)
-> build_loss_transform_pipeline(...)
```

4. Updated acoustic/elastic drivers and focused tests to use the new function
   name.

## Numerical Scope

This is a readability-only change. It does not change:

- transform order;
- receiver selection order;
- waveform normalization;
- misfit dispatch behavior;
- component weighting;
- regularization merge;
- backward/update order.

## Validation

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/fwi/iteration/loss.py \
  ADFWI/fwi/iteration/step.py \
  ADFWI/fwi/acoustic_fwi.py \
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

Review `iteration/loss.py` as a human-facing file. If another adjustment is
needed, prefer local naming/docstring changes over creating more helper modules.
