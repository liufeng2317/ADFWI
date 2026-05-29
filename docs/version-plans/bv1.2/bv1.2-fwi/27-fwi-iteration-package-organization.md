# FWI Iteration Package Organization

## Goal

Continue reducing flat helper modules under `ADFWI.fwi` by making the shared iteration helpers follow the same package style as `fwi.data`, `fwi.misfit`, `fwi.optimizer`, `fwi.regularization`, and `fwi.transforms`.

## Change

`ADFWI.fwi.iteration` was converted from a single module into a package with responsibility-focused modules:

- `ADFWI.fwi.iteration.range`: `BatchRange` and `iter_batch_ranges`.
- `ADFWI.fwi.iteration.loss`: `BatchLoss` and `build_batch_loss`.
- `ADFWI.fwi.iteration.progress`: `set_batch_description`.

The package-level `ADFWI.fwi.iteration.__init__` re-exports the existing public helper names, so current imports such as `from ADFWI.fwi.iteration import iter_batch_ranges` continue to work.

## Numerical Contract

This is an import-organization refactor only. Batch slicing rules, scalar loss bookkeeping, and progress label behavior were moved without changing their logic. Propagation, misfit evaluation, transforms, regularization, and optimizer steps are untouched.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/iteration/__init__.py ADFWI/fwi/iteration/range.py ADFWI/fwi/iteration/loss.py ADFWI/fwi/iteration/progress.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_iteration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration.py tests/test_fwi_data_contract.py tests/test_backend_integration.py` passed: 53 tests.
- `conda run -n adfwi python scripts/smoke/compare_backend_smoke.py --problems acoustic,elastic --cases trace-missing --devices cpu,npu:0` passed.

CPU/NPU smoke comparison:

| Problem | Metric | Relative difference | Status |
| --- | --- | ---: | --- |
| acoustic | loss | 0.0 | pass |
| acoustic | vp_grad_norm | 0.0 | pass |
| acoustic | vp_update_norm | 0.0 | pass |
| elastic | loss | 4.880620563312549e-06 | pass |
| elastic | vp_grad_norm | 1.2590211728446073e-06 | pass |
| elastic | vp_update_norm | 0.0 | pass |

The smoke results match the previous bv1.2 baseline range, confirming the iteration package organization pass does not change FWI numerical behavior.
