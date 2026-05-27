# 102. Data Public Facade

## Goal

Keep `ADFWI.fwi.data` as a stable public facade for FWI data-contract helpers
while keeping framework internals on responsibility-focused owner modules.

`ADFWI.fwi.data` differs from `ADFWI.fwi.iteration` and
`ADFWI.fwi.runtime`: it describes user-visible concepts such as synthetic and
observed waveform pairing, transform context construction, misfit dispatch, and
elastic component bookkeeping. A small facade makes the framework easier to use
without hiding internal ownership from production FWI drivers.

## Optimization Path

1. Update `ADFWI/fwi/data/__init__.py` docstring from a historical compatibility
   surface to a deliberate public data-contract facade.
2. Keep current data-contract exports stable and group `__all__` by role:
   records/constants, preparation, loss input/evaluation, and elastic component
   helpers.
3. Add `tests/test_fwi_data_public_api.py` to lock the facade:
   - The facade exports exactly the curated public symbols.
   - Each facade symbol points to its owner-module implementation object.

## Numerical Contract

No numerical implementation changed. The facade still re-exports existing
objects from owner modules:

- `data.components`
- `data.inputs`
- `data.loss`
- `data.pipeline`
- `data.preparation`

The test verifies object identity rather than recomputing numerical outputs,
because this change only defines import ownership.

## Validation Result

Focused public API, data-contract, runtime, iteration, and transform tests
passed:

```bash
conda run -n adfwi python -m unittest tests/test_fwi_data_public_api.py tests/test_fwi_data_contract.py tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_data_transforms.py
# Ran 86 tests in 12.763s, OK
```

Import and syntax checks passed:

```bash
conda run -n adfwi python -m py_compile ADFWI/fwi/data/__init__.py tests/test_fwi_data_public_api.py
# OK
```

Active FWI internals still avoid the facade:

```bash
rg -n "from ADFWI\\.fwi\\.data import|ADFWI\\.fwi\\.data import" ADFWI scripts tests examples --glob '!tests/test_fwi_data_public_api.py' --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
# no matches
```

## Next Direction

Document the recommended user-facing import style:

```python
from ADFWI.fwi.data import prepare_fwi_loss_pair, build_fwi_data_transform_pipeline
```

Keep internal FWI drivers on owner-module imports.
