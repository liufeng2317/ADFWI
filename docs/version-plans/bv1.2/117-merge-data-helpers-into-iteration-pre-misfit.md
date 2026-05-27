# Merge Data Helpers Into Iteration Pre-Misfit Modules

## Goal

Remove the misleading `ADFWI.fwi.data` package from `bv1.2`.

The helpers under `fwi/data` were not data loaders or dataset abstractions. They
were the small observation-preparation part of the FWI iteration loop:

- acoustic/elastic component selection;
- synthetic/observed pair records;
- receiver/data-mask and transform-context preparation;
- default pre-misfit transform pipeline construction;
- misfit dispatch and weighted component-loss summation.

Keeping these helpers under `data` made the framework boundary look broader
than it really was and overlapped with the real mathematical losses in
`ADFWI.fwi.misfit`.

## Change

Moved the previous `ADFWI.fwi.data` modules into iteration-owned pre-misfit
modules:

```text
ADFWI/fwi/iteration/components.py
ADFWI/fwi/iteration/pairs.py
ADFWI/fwi/iteration/preparation.py
ADFWI/fwi/iteration/misfit.py
```

Removed:

```text
ADFWI/fwi/data/
tests/test_fwi_data_public_api.py
```

Renamed the focused contract test:

```text
tests/test_fwi_data_contract.py
-> tests/test_fwi_iteration_pre_misfit.py
```

Updated active imports in:

- `ADFWI/fwi/acoustic_fwi.py`
- `ADFWI/fwi/elastic_fwi.py`
- `ADFWI/fwi/iteration/loss.py`

The import-surface policy now treats `ADFWI.fwi.data` as removed.

## Numerical Scope

This is a structure-only migration. It does not change:

- receiver selection order;
- transform order;
- waveform normalization formula;
- misfit dispatch rules;
- component weighting;
- regularization addition;
- backward/update ordering.

Because no FWI numerical logic changed, validation focuses on import policy,
observation-preparation parity through the migrated tests, and existing
iteration/runtime tests.

## Validation Plan

```bash
conda run -n adfwi python -m unittest \
  tests/test_fwi_iteration_pre_misfit.py \
  tests/test_fwi_iteration.py \
  tests/test_fwi_runtime.py \
  tests/test_import_surface_policy.py

conda run -n adfwi python -m py_compile \
  ADFWI/fwi/iteration/components.py \
  ADFWI/fwi/iteration/pairs.py \
  ADFWI/fwi/iteration/preparation.py \
  ADFWI/fwi/iteration/misfit.py \
  ADFWI/fwi/iteration/loss.py \
  ADFWI/fwi/acoustic_fwi.py \
  ADFWI/fwi/elastic_fwi.py
```

## Next Direction

Do not recreate a separate data facade in `bv1.2`. If this area needs more
cleanup, keep the stages aligned with the FWI iteration flow: component
selection, pair construction, pre-misfit preparation, misfit evaluation, then
batch optimization.
