# FWI Data Package Organization

## Goal

Reduce the flat-helper feel under `ADFWI.fwi` and make the shared FWI data path follow the same layered style as `fwi.misfit`, `fwi.optimizer`, `fwi.regularization`, and `fwi.transforms`.

## Change

`ADFWI.fwi.data` was converted from a single module into a package with responsibility-focused modules:

- `ADFWI.fwi.data.pipeline`: default FWI transform pipeline construction.
- `ADFWI.fwi.data.preparation`: transform context construction and synthetic/observed pair preparation.
- `ADFWI.fwi.data.loss`: misfit dispatch and weighted loss summation helpers.
- `ADFWI.fwi.data.components`: elastic component naming, pressure construction, component input selection, and component weight validation.

The package-level `ADFWI.fwi.data.__init__` re-exports the existing public helper names, so current imports such as `from ADFWI.fwi.data import prepare_fwi_loss_pair` continue to work.

## Numerical Contract

This is an import-organization refactor only. Function bodies were moved into responsibility-focused modules without changing formulas, transform order, receiver selection, component ordering, misfit dispatch, or loss accumulation.

## Validation

Completed on 2026-05-26:

- `python -m py_compile ADFWI/fwi/data/__init__.py ADFWI/fwi/data/pipeline.py ADFWI/fwi/data/preparation.py ADFWI/fwi/data/loss.py ADFWI/fwi/data/components.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_data_contract.py tests/test_backend_integration.py` passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_backend_integration.py tests/test_fwi_iteration.py` passed: 53 tests.
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

The smoke results match the previous bv1.2 baseline range, confirming the package organization pass does not change FWI numerical behavior.
