# 101. Data Owner Imports

## Goal

Route active FWI data-contract callers to responsibility-focused owner modules
instead of the broad `ADFWI.fwi.data` aggregation layer. Unlike `iteration` and
`runtime`, the data package may remain a user-facing contract API, so this step
only removes active internal dependence on the aggregation surface.

## Optimization Path

1. Move `AcousticFWI` data imports to owner modules:
   - `data.pipeline` for default transform pipeline construction.
   - `data.preparation` for transform context and pre-loss pair preparation.
   - `data.loss` for misfit dispatch.
2. Move `ElasticFWI` data imports to the same owner modules, plus
   `data.components` for observed component assembly and component weights.
3. Move `tests/test_fwi_data_contract.py` to owner-module imports so focused
   tests validate the canonical implementation modules directly.
4. Confirm active code, tests, examples, and scripts no longer import helpers
   through `from ADFWI.fwi.data import ...`.

## Numerical Contract

This change does not alter FWI data-contract helper implementations. It only
changes import locations for existing functions and dataclasses. Therefore the
following behavior is unchanged:

- receiver selection before transform pipelines.
- transform context construction.
- legacy-compatible default transform pipeline construction.
- acoustic pressure and elastic component loss-input selection.
- misfit dispatch and weighted component summation.

## Validation Result

Focused data-contract, runtime, iteration, and transform tests passed after
fixing one test-import omission for `elastic_component_loss_inputs`:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_data_transforms.py
# Ran 84 tests in 11.857s, OK
```

Import and syntax checks passed:

```bash
conda run -n adfwi python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py ADFWI/fwi/data/__init__.py ADFWI/fwi/data/components.py ADFWI/fwi/data/inputs.py ADFWI/fwi/data/loss.py ADFWI/fwi/data/pipeline.py ADFWI/fwi/data/preparation.py
# OK
```

Active code no longer uses the data aggregation surface:

```bash
rg -n "from ADFWI\\.fwi\\.data import|ADFWI\\.fwi\\.data import" ADFWI scripts tests examples --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
# no matches
```

## Next Direction

Decide separately whether `ADFWI.fwi.data.__init__` should remain as a stable
user-facing data-contract API. If it stays public, update docs to present it as
the supported user import surface while keeping internal FWI drivers on owner
modules.
