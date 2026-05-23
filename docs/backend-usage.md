# ADFWI Backend Usage

This document records the bv1.2 user-facing backend API for CPU, CUDA GPU, and NPU execution.

## One-Line Backend Setup

For normal scripts and notebooks, configure the framework backend once before creating models, propagators, regularization objects, or FWI engines:

```python
import ADFWI

ADFWI.set_backend("npu:0")
```

After this call, core ADFWI objects that accept `device=None` inherit the active backend by default.

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

Backend resolution and integration tests:

```bash
conda run -n adfwi python -m unittest tests/test_backends.py tests/test_backend_integration.py
```

Acoustic smoke tests:

```bash
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device npu:0
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2
```

Elastic smoke test:

```bash
conda run -n adfwi python scripts/smoke/elastic_backend_smoke.py --device npu:0
```

Misfit smoke tests:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0
```
