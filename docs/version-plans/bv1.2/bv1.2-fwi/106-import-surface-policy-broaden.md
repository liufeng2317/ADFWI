# 106. Broaden Import Surface Policy

## Goal

Broaden the tracked-file import surface policy so namespace-only packages cannot
return through alternate import syntax.

## Optimization Path

The initial policy test blocked direct broad imports such as:

```python
from ADFWI.fwi.runtime import ...
from ADFWI.fwi.iteration import ...
```

This pass also blocks equivalent package-level imports:

```python
import ADFWI.fwi.runtime
import ADFWI.fwi.iteration
from ADFWI.fwi import runtime
from ADFWI.fwi import iteration
```

The data facade policy is also broadened to catch:

```python
import ADFWI.fwi.data
```

outside the allowed public API documentation and facade test.

## Numerical Contract

Test-only policy change. No FWI runtime, data-contract, gradient, wavefield, or
propagator code changed.

## Validation Result

The first validation run exposed a self-scan issue: the policy test was matching
its own marker definitions. The test now skips `tests/test_import_surface_policy.py`
when scanning tracked files.

The broadened policy test passed:

```bash
conda run -n adfwi python -m unittest tests/test_import_surface_policy.py
# Ran 2 tests in 2.932s, OK
```

The focused public facade/low-pass group also passed:

```bash
conda run -n adfwi python -m unittest tests/test_import_surface_policy.py tests/test_fwi_data_public_api.py tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py
# Ran 11 tests in 2.855s, OK
```

Syntax check passed:

```bash
conda run -n adfwi python -m py_compile tests/test_import_surface_policy.py
# OK
```

## Next Direction

Use this broadened policy test as part of lightweight validation whenever
examples, docs, or FWI package imports are touched.
