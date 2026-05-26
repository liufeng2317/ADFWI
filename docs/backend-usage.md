# ADFWI Backend Usage

This document records the bv1.2 user-facing backend API for CPU, CUDA GPU, and NPU execution.

## Recommended Public Entry Points

For research scripts and notebooks, prefer the top-level `ADFWI` API:

```python
import ADFWI

ADFWI.set_backend("npu:0", dtype="float32")
print(ADFWI.backend_diagnostics())
```

Use `ADFWI.backends` when writing tests, smoke scripts, or framework internals that need lower-level helpers such as `resolve_backend(...)` or `use_backend(...)`.

| User need | Recommended API |
| --- | --- |
| Configure one global device for a notebook/script | `ADFWI.set_backend(...)` |
| Inspect the active backend | `ADFWI.backend()` or `ADFWI.backend_diagnostics()` |
| Temporarily override backend in tests | `from ADFWI.backends import use_backend` |
| Resolve a backend without changing global state | `from ADFWI.backends import resolve_backend` |

## One-Line Backend Setup

For normal scripts and notebooks, configure the framework backend once before creating models, propagators, regularization objects, or FWI engines:

```python
import ADFWI

ADFWI.set_backend("npu:0")
```

After this call, core ADFWI objects that accept `device=None` inherit the active backend by default. Create objects in this order when possible:

1. call `ADFWI.set_backend(...)`;
2. create models and propagators;
3. create regularization and FWI drivers;
4. run forward modeling or inversion.

This keeps model tensors, propagator buffers, regularization tensors, and FWI runtime checks on one consistent device/dtype.

Equivalent lower-level API:

```python
from ADFWI.backends import set_backend, backend

set_backend("npu:0")
print(backend().diagnostics())
```

## Automatic Selection

Use `device=None` to let ADFWI choose from a priority list:

```python
ADFWI.set_backend(None, prefer="npu,cpu")
ADFWI.set_backend(None, prefer="cuda,cpu")
ADFWI.set_backend(None, prefer="cuda,npu,cpu")
```

The default priority is `npu,cpu`, matching the current local `adfwi` CPU+NPU environment. The priority can be passed either as a comma-separated string or as a tuple/list:

```python
ADFWI.set_backend(None, prefer=("npu", "cpu"))
```

## Dtype

The backend dtype can be passed as either a PyTorch dtype or a string:

```python
ADFWI.set_backend("cpu", dtype="float32")
ADFWI.set_backend("cpu", dtype="float64")
```

Supported string aliases are `float16`, `half`, `float32`, `float64`, and `double`.

## Diagnostics

Use diagnostics in logs, smoke tests, and benchmark metadata:

```python
info = ADFWI.backend_diagnostics()
print(info)
```

Typical fields:

```text
name, device, index, dtype, available, fallback, reason, memory_allocated
```

## Scoped Overrides

For tests or small mixed-device experiments, use a temporary backend context:

```python
from ADFWI.backends import use_backend

ADFWI.set_backend("npu:0")

with use_backend("cpu"):
    # objects created here inherit CPU
    ...

# previous backend is restored here
```

## Explicit Object Overrides

Existing `device=` arguments remain supported during bv1.2 migration:

```python
model = AcousticModel(..., device="cpu")
```

Use explicit object overrides only when intentionally mixing devices or keeping legacy scripts unchanged. New examples should prefer one framework-level backend setup call.

## Smoke Test Commands

Backend public API and integration tests:

```bash
conda run -n adfwi python -m unittest tests/test_backends.py tests/test_backend_integration.py
```

Lightweight public API smoke tests:

```bash
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device auto --prefer npu,cpu
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device cpu --dtype float64
conda run -n adfwi python scripts/smoke/backend_public_api_smoke.py --device npu:0
```

Acoustic smoke tests:

```bash
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device npu:0
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2
```

Elastic smoke tests:

```bash
conda run -n adfwi python scripts/smoke/elastic_backend_smoke.py --device npu:0
conda run -n adfwi python scripts/smoke/elastic_mini_inversion_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/elastic_mini_inversion_smoke.py --device npu:0
```

Misfit smoke tests:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0
```


## Public API Stability

The researcher-facing top-level API is intentionally small:

```python
import ADFWI

ADFWI.set_backend(...)
ADFWI.get_backend()
ADFWI.backend()
ADFWI.backend_diagnostics()
```

The lower-level `ADFWI.backends` package additionally exports `Backend`, backend-specific errors, `configure_backend`, `resolve_backend`, and `use_backend`. Tests pin these `__all__` exports so future cleanup does not accidentally remove the documented entry points.
