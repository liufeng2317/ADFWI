# FWI Iteration Epoch Finalization Helper

## Goal

Move repeated epoch-final cache gating and progress text updates out of
`AcousticFWI` and `ElasticFWI` while keeping model-specific result saving in each
FWI driver.

## Change

- Added `ADFWI.fwi.iteration.finalize_epoch_progress`.
- Updated `AcousticFWI.forward()` and `AcousticFWI.forward_closure()` to use the
  helper after model constraints are applied.
- Updated `ElasticFWI.forward()` to use the helper after model constraints are
  applied.
- Added focused tests for cache callback gating and legacy epoch progress text.

## Scientific Contract

This is a structural finalization refactor. It must preserve:

- cache execution after optimizer, scheduler, and model constraint steps;
- the `cache_result` gate;
- the exact `save_model_and_gradients(epoch_id=..., loss_epoch=...)` callback
  contract;
- legacy epoch progress label formatting: `Iter:{},Loss:{:.4}`;
- model-specific cached parameter choices and figure-saving behavior inside each
  FWI driver.

## Validation

Completed on 2026-05-26.

Syntax check:

```bash
python -m py_compile ADFWI/fwi/iteration/progress.py ADFWI/fwi/iteration/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py
```

Result: passed.

Focused iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_iteration.py
```

Result: passed, 14 tests.

Broader runtime/backend/data/iteration tests:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_backend_integration.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py
```

Result: passed, 75 tests.

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

The next cleanup should focus on removing unused per-batch local variables in
FWI loops, such as `begin_index` and `end_index` where the structured
`BatchRange` already carries the same information. This is a low-risk readability
pass that should keep all physical data selection through `shot_index`.
