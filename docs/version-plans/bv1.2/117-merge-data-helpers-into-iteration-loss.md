# Merge Data Helpers Into Iteration Loss

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

Moved the previous `ADFWI.fwi.data` modules into the iteration-owned loss
construction module:

```text
ADFWI/fwi/iteration/loss.py
```

One-batch forward/loss/backward execution is separated into:

```text
ADFWI/fwi/iteration/step.py
```

Removed:

```text
ADFWI/fwi/data/
tests/test_fwi_data_public_api.py
```

Renamed the focused contract test:

```text
tests/test_fwi_data_contract.py
-> tests/test_fwi_iteration_loss.py
```

Updated active imports in:

- `ADFWI/fwi/acoustic_fwi.py`
- `ADFWI/fwi/elastic_fwi.py`
- `ADFWI/fwi/iteration/loss.py`
- `ADFWI/fwi/iteration/step.py`

The import-surface policy now treats `ADFWI.fwi.data` as removed.
It also guards against recreating the short-lived split modules
`ADFWI.fwi.iteration.components`, `pairs`, `preparation`, and `misfit`.

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
  tests/test_fwi_iteration_loss.py \
  tests/test_fwi_iteration.py \
  tests/test_fwi_runtime.py \
  tests/test_import_surface_policy.py

conda run -n adfwi python -m py_compile \
  ADFWI/fwi/iteration/loss.py \
  ADFWI/fwi/iteration/step.py \
  ADFWI/fwi/iteration/batches.py \
  ADFWI/fwi/iteration/epoch.py \
  ADFWI/fwi/acoustic_fwi.py \
  ADFWI/fwi/elastic_fwi.py
```

## Next Direction

Do not recreate a separate data facade in `bv1.2`. If this area needs more
cleanup, keep it inside `ADFWI.fwi.iteration.loss` unless a measured complexity
problem appears. Shot batch scheduling belongs to `ADFWI.fwi.iteration.batches`.
