# 35. User-Facing Backend Interface

## Goal

Stabilize the public backend/device entry points after the bv1.2 backend and runtime cleanup. The objective is to make one-line CPU/NPU/CUDA setup clear for geophysical researchers while preserving lower-level migration helpers for tests and framework internals.

## Public Interface Decision

For normal research scripts and notebooks, the recommended entry point is the top-level package API:

```python
import ADFWI

ADFWI.set_backend("npu:0", dtype="float32")
print(ADFWI.backend_diagnostics())
```

The lower-level package remains available for advanced usage:

```python
from ADFWI.backends import resolve_backend, use_backend
```

## Changes

- Added public API stability tests for `ADFWI.__all__` and `ADFWI.backends.__all__`.
- Updated `docs/backend-usage.md` with recommended researcher-facing and lower-level backend entry points.
- Added a short backend setup pointer to the root `README.md` so users can find the device guide without reading the bv1.2 planning notes.

## Boundary

This update does not change backend resolution behavior, default device priority, tensor creation, model construction, propagator behavior, or FWI numerical paths. It only makes the public entry points and usage order explicit.

## Validation

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile tests/test_backends.py`: passed.
- `python -m unittest tests/test_backends.py tests/test_backend_integration.py`: 38 tests passed, 1 skipped.

No acoustic/elastic smoke comparison was required because this update only changes public API tests and documentation; backend resolution and numerical execution paths were not changed.
