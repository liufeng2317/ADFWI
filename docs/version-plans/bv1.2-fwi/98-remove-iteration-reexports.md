# 98. Remove Iteration Re-Exports

## Goal

Finish the iteration package cleanup started in record 97 by removing broad
package-level helper re-exports from `ADFWI.fwi.iteration`. In bv1.2 the
iteration package is a namespace, and helpers are imported from their owner
modules.

## Optimization Path

1. Confirm active code, tests, examples, and scripts no longer import helpers
   through `from ADFWI.fwi.iteration import ...`.
2. Remove the package-level imports and `__all__` list from
   `ADFWI/fwi/iteration/__init__.py`.
3. Keep a short package docstring that points users to the canonical owner
   modules:
   - `ADFWI.fwi.iteration.range`
   - `ADFWI.fwi.iteration.loss`
   - `ADFWI.fwi.iteration.progress`
   - `ADFWI.fwi.iteration.epoch`

## Numerical Contract

This change does not modify FWI execution code. It only removes an import
aggregation layer. No tensor operation, propagator call, loss expression,
backward call, optimizer step, scheduler step, or progress mutation behavior is
changed.

## Validation Result

Run the same focused iteration/runtime validation in the `adfwi` environment:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py
# Ran 89 tests in 11.429s, OK
```

Run import and syntax checks for the iteration package and FWI drivers:

```bash
conda run -n adfwi python -m py_compile ADFWI/fwi/iteration/__init__.py ADFWI/fwi/iteration/range.py ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/progress.py ADFWI/fwi/iteration/epoch.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py
# OK
```

Owner-module import smoke:

```bash
conda run -n adfwi python -c 'from ADFWI.fwi.iteration.range import iter_batch_ranges; from ADFWI.fwi.iteration.loss import build_batch_loss, apply_batch_loss_step; from ADFWI.fwi.iteration.progress import finalize_epoch_progress; from ADFWI.fwi.iteration.epoch import apply_epoch_update_step; print([(r.batch, r.begin, r.end, r.shot_index.tolist()) for r in iter_batch_ranges(5, 2)])'
# [(0, 0, 2, [0, 1]), (1, 2, 4, [2, 3]), (2, 4, 5, [4])]
```

Because the changed file is an import surface only, numerical drift is exactly
zero by construction; the focused tests cover the helper behavior through owner
module imports.

## Next Direction

Continue the same staged cleanup for other bv1.2 package-level aggregation
surfaces. Prioritize active FWI internals first, and keep explicit legacy
numerical methods in place until a focused numerical comparison accepts a
replacement.
