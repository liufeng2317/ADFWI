# 100. Remove Runtime Re-Exports

## Goal

Finish the runtime package cleanup started in record 99 by removing broad
package-level helper re-exports from `ADFWI.fwi.runtime`. In bv1.2 the runtime
package is a namespace, and helpers are imported from responsibility-focused
owner modules.

## Optimization Path

1. Confirm active code, tests, examples, and scripts no longer import helpers
   through `from ADFWI.fwi.runtime import ...`.
2. Remove package-level imports and `__all__` from
   `ADFWI/fwi/runtime/__init__.py`.
3. Keep a short package docstring that documents canonical owner modules:
   - `ADFWI.fwi.runtime.backend`
   - `ADFWI.fwi.runtime.cache`
   - `ADFWI.fwi.runtime.forward`
   - `ADFWI.fwi.runtime.gradient`
   - `ADFWI.fwi.runtime.regularization`
   - `ADFWI.fwi.runtime.wavefield`

## Numerical Contract

This change removes an import aggregation layer only. It does not modify any
runtime helper implementation. FWI behavior is unchanged for:

- backend/device validation.
- result-cache bookkeeping.
- per-batch forward records.
- gradient parameter specs and gradient processing.
- regularization loss calculation.
- wavefield selection and accumulation.

## Validation Result

Focused runtime, iteration, data-contract, and torch-gradient tests passed:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_fwi_data_contract.py tests/test_torch_grad_processor.py
# Ran 71 tests in 0.224s, OK
```

Import and syntax checks passed:

```bash
conda run -n adfwi python -m py_compile ADFWI/fwi/runtime/__init__.py ADFWI/fwi/runtime/backend.py ADFWI/fwi/runtime/cache.py ADFWI/fwi/runtime/forward.py ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/regularization.py ADFWI/fwi/runtime/wavefield.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py ADFWI/fwi/iteration/loss.py
# OK
```

Active code no longer uses the removed aggregation surface:

```bash
rg -n "from ADFWI\\.fwi\\.runtime import|ADFWI\\.fwi\\.runtime import" ADFWI scripts tests examples --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
# no matches
```

## Next Direction

Audit `ADFWI.fwi.data` next. Unlike `runtime`, the data package is closer to a
user-facing contract layer, so only remove its aggregation exports after the
recommended public API is clear.
