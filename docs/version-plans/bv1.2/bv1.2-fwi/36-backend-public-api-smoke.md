# 36. Backend Public API Smoke

## Goal

Add a lightweight smoke script for the user-facing backend API. This gives researchers a fast way to check CPU/NPU/CUDA backend setup before running acoustic or elastic propagation tests.

## Scope

Added `scripts/smoke/backend_public_api_smoke.py`.

The script checks:

- top-level `ADFWI.set_backend(...)` configuration;
- `ADFWI.backend_diagnostics()` JSON-friendly diagnostics;
- backend tensor factory dtype/device behavior;
- temporary `use_backend("cpu")` scoped override and restoration.

## Boundary

This smoke test does not run wave propagation, FWI, misfit evaluation, or file output. It is an environment/API entry-point check only.

## Usage

```bash
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device auto --prefer npu,cpu
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device cpu --dtype float64
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device npu:0
```

## Validation

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile scripts/smoke/backend_public_api_smoke.py`: passed.
- `conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device cpu --dtype float64`: passed, resolved `cpu`, dtype `float64`.
- `conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device auto --prefer npu,cpu`: passed, resolved `npu:0`, dtype `float32`.
- `conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device npu:0`: passed, resolved `npu:0`, dtype `float32`.

All runs verified tensor factory dtype/device behavior and restoration after a scoped CPU override.
