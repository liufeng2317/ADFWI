# 99. Runtime Owner Imports

## Goal

Continue the bv1.2 import-surface cleanup by routing active FWI runtime users
to responsibility-focused owner modules instead of the broad
`ADFWI.fwi.runtime` aggregation layer.

## Optimization Path

1. Move `AcousticFWI` runtime imports to owner modules:
   - `runtime.backend` for backend/device validation.
   - `runtime.cache` for loss/model/gradient snapshot bookkeeping.
   - `runtime.gradient` for parameter name/spec helpers and gradient dispatch.
   - `runtime.regularization` for regularization loss helpers.
2. Move `ElasticFWI` runtime imports to the same owner modules, plus
   `runtime.wavefield` for elastic gradient-wavefield selection.
3. Move `ADFWI.fwi.iteration.loss` imports to `runtime.forward` and
   `runtime.wavefield`, matching the actual helper ownership.
4. Update runtime and torch-gradient tests so active tests no longer depend on
   `from ADFWI.fwi.runtime import ...`.

## Numerical Contract

This change does not alter runtime helper implementations. It only changes
where active callers import the existing functions from. Therefore the FWI
execution contract is unchanged:

- model/propagator device validation is still in `runtime.backend`.
- regularization loss formulas are still in `runtime.regularization`.
- gradient processing still preserves the legacy NumPy/Torch processor paths in
  `runtime.gradient`.
- forward batch records and wavefield accumulation still use the same runtime
  helpers.

## Validation Result

Focused runtime, iteration, data-contract, and torch-gradient tests passed:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_fwi_data_contract.py tests/test_torch_grad_processor.py
# Ran 71 tests in 0.233s, OK
```

Import and syntax checks for FWI drivers and runtime modules passed:

```bash
conda run -n adfwi python -m py_compile ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py ADFWI/fwi/iteration/loss.py ADFWI/fwi/runtime/backend.py ADFWI/fwi/runtime/cache.py ADFWI/fwi/runtime/forward.py ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/regularization.py ADFWI/fwi/runtime/wavefield.py ADFWI/fwi/runtime/__init__.py
# OK
```

Active code no longer imports helpers from the runtime aggregation layer:

```bash
rg -n "from ADFWI\\.fwi\\.runtime import|ADFWI\\.fwi\\.runtime import" ADFWI scripts tests examples --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
# no matches
```

## Next Direction

If validation remains clean, decide in a separate step whether
`ADFWI.fwi.runtime.__init__` should remain a public aggregation layer or become
a namespace-only package like `ADFWI.fwi.iteration`.
