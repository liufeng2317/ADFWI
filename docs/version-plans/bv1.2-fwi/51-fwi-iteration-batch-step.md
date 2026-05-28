# FWI Iteration Batch Step Helper

## Goal

Reduce repeated per-batch bookkeeping in `AcousticFWI` and `ElasticFWI` while
keeping the geophysical workflow readable. The FWI drivers still construct the
synthetic/observed pairs, component losses, and optional model regularization;
the iteration helper only applies the common batch step.

## Change

- Added `ADFWI.fwi.iteration.apply_batch_loss_step`.
- Updated `AcousticFWI.forward()` and `AcousticFWI.forward_closure()` to use the
  helper after acoustic data loss and optional regularization are computed.
- Updated `ElasticFWI.forward()` to use the helper after elastic component loss
  aggregation and optional regularization are computed.
- Added focused iteration tests for scalar accumulation, backward execution,
  optional regularization, and progress description behavior.

## Scientific Contract

This is a structural iteration refactor. It must preserve:

- data-loss construction in the acoustic/elastic drivers;
- elastic component selection and weighting before the shared helper is called;
- optional regularization inclusion rule;
- scalar epoch loss accumulation using the historical `.item()` values;
- exactly one backward call per batch loss;
- legacy single-batch progress description behavior;
- optimizer, scheduler, gradient processing, model constraint, cache, and plotting
  order.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py
```

Result: passed.

Focused iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_iteration.py
```

Result: passed, 10 tests.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 71 tests.

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

The next cleanup should target epoch-level optimizer/scheduler/model-constraint
bookkeeping. A small helper can apply `optimizer.step()`, optional closure
handling where appropriate, `scheduler.step()`, and `model.forward()` ordering.
Keep LBFGS/NLCG closure construction inside `AcousticFWI` for now, because that
control flow is optimizer-specific and more delicate than the per-batch step.
