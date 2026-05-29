# FWI Iteration Epoch Update Helper

## Goal

Move the repeated epoch-tail update sequence out of `AcousticFWI` and
`ElasticFWI` while keeping optimizer-specific closure construction in the FWI
drivers. The helper preserves the legacy order: optimizer step first, scheduler
step second, model constraint last.

## Change

- Added `ADFWI.fwi.iteration.apply_epoch_update_step`.
- Updated `AcousticFWI.forward()` to use the helper for non-closure optimizers.
- Updated `AcousticFWI.forward_closure()` to pass its existing closure into the
  helper and preserve the optimizer return value.
- Updated `ElasticFWI.forward()` to use the helper after gradient processing.
- Added focused iteration tests for update order and closure return behavior.

## Scientific Contract

This is a structural epoch-tail refactor. It must preserve:

- all loss construction, backward calls, and gradient processing before the
  optimizer step;
- `optimizer.step()` before `scheduler.step()`;
- `scheduler.step()` before `model.forward()` bounds/constraint enforcement;
- closure construction inside `AcousticFWI.forward_closure()`;
- the value returned by closure-based optimizer steps;
- cache and progress updates after model constraints are applied.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/iteration/epoch.py ADFWI/fwi/iteration/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py
```

Result: passed.

Focused iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_iteration.py
```

Result: passed, 12 tests.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 73 tests.

Acoustic/elastic CPU/NPU smoke comparison:

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0
```

Result: passed.

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 0.0 | pass |
| acoustic | vp_grad_norm | 0.0 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.880620563312549e-06 | pass |
| elastic | vp_grad_norm | 1.2590211728446073e-06 | pass |
| elastic | vp_update_norm | 0.0 | pass |

The elastic CPU/NPU drift remains inside the established `1e-5` relative
tolerance and matches the previous bv1.2 backend smoke behavior.

## Next Optimization Direction

The next cleanup should target epoch progress/cache finalization. A helper could
handle cache-result gating and epoch progress text, but figure-saving and cached
parameter choices should remain in each FWI driver because they are tied to
acoustic versus elastic model semantics.
