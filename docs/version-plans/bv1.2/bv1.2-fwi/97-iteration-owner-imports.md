# 97. Iteration Owner Imports

## Goal

Continue the bv1.2 compatibility cleanup by routing FWI iteration users to the
module that owns each helper instead of the package-level aggregation surface.
This prepares `ADFWI.fwi.iteration.__init__` for a later public API narrowing
without changing FWI numerical behavior.

## Optimization Path

1. Move `AcousticFWI` and `ElasticFWI` imports from `ADFWI.fwi.iteration` to
   owner modules:
   - `ADFWI.fwi.iteration.range` for batch ranges.
   - `ADFWI.fwi.iteration.loss` for acoustic/elastic batch loss steps.
   - `ADFWI.fwi.iteration.epoch` for optimizer/scheduler/model update order.
   - `ADFWI.fwi.iteration.progress` for epoch progress finalization.
2. Move iteration tests to the same owner-module imports so the active test
   suite no longer depends on broad package-level re-exports.
3. Hoist the `set_batch_description` import out of
   `apply_batch_loss_step`. The helper is now resolved once when
   `ADFWI.fwi.iteration.loss` is imported instead of during every batch-loss
   application.

## Numerical Contract

No numerical formula, tensor operation, propagator call, loss construction, or
backward call was changed. The batch-loss helper still computes:

- `data_loss.item()` when no regularization is present.
- `data_loss.item() + regularization_loss.item()` when regularization is
  present.
- `(data_loss + regularization_loss).backward()` for the tensor path.

The pre-change deterministic scalar/gradient reference was:

```json
{"epoch_loss": 10.0, "grad": [1.5, -2.0, 5.5], "loss": 4.5, "reg": 3.5}
```

The same reference should remain bitwise identical after the change.

## Validation Result

Focused unit and import checks passed in the `adfwi` environment:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py
# Ran 89 tests in 11.301s, OK

conda run -n adfwi python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/progress.py ADFWI/fwi/iteration/range.py ADFWI/fwi/iteration/epoch.py
# OK
```

Because this touches the FWI iteration helper, the deterministic loss/gradient
reference was rerun after the change:

```json
{"epoch_loss": 10.0, "grad": [1.5, -2.0, 5.5], "loss": 4.5, "reg": 3.5}
```

The post-change values match the pre-change reference exactly.

## Next Direction

Audit whether `ADFWI.fwi.iteration.__init__` should remain a supported public
aggregation layer in bv1.2. If active examples and user-facing docs do not need
it, narrow or remove the broad re-exports in a separate commit with explicit
import-breakage notes.
