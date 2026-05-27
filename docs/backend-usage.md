# ADFWI Backend Usage

This document records the bv1.2 user-facing backend API for CPU, CUDA GPU, and NPU execution.

For the broader bv1.2 optimization map, see
[ADFWI bv1.2 Optimization Chain](./version-plans/bv1.2/optimization-chain.md).

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

## Minimal Script Example

A small script-style example is available for users who want to copy the bv1.2 backend pattern without editing notebook outputs:

```bash
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device auto --prefer npu,cpu
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device npu:0
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device npu:0
```

The acoustic example configures `ADFWI.set_backend(...)`, builds a tiny acoustic true/initial model pair, synthesizes observed data in memory, runs one AcousticFWI iteration, and prints a JSON summary. The elastic example mirrors the same pattern with an isotropic ElasticFWI pressure-component inversion.

See `scripts/examples/README.md` for the researcher-facing example guide and migration notes for adapting the minimal scripts toward real cases.

A read-only Marmousi2 acoustic case check is also available:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device cpu
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device npu:0
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300
```

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

Layered backend smoke suite:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,misfit --devices cpu,npu:0 --misfits L2,SquaredL2,StudentT
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic --example-gradient-processors legacy,torch
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic --case-run-forward --case-shot-index 0
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices npu:0 --case-inversion-iterations 10
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,compare-mini --devices cpu,npu:0 --compare-cases trace-missing
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

Benchmark legacy versus torch gradient processing:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_backend_benchmark.py --device cpu --warmup 1 --repeat 3 --gradient-processors legacy,torch
conda run -n adfwi python scripts/benchmark/acoustic_backend_benchmark.py --device npu:0 --warmup 1 --repeat 3 --gradient-processors legacy,torch
```

Opt-in full-case forward-plus-inversion test:

```bash
conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
ADFWI_RUN_FULL_CASES=1 ADFWI_FULL_CASE_OUTPUT_DIR=tests/full_cases/outputs/marmousi2_latest conda run -n adfwi python -m unittest tests/full_cases/test_marmousi2_acoustic_full_flow.py
```


## FWI Data Contract API

Use `ADFWI.fwi.data` as the stable public entry point when custom scripts need
to prepare synthetic/observed waveform pairs before a misfit, build the default
FWI transform pipeline, or configure elastic component weights:

```python
from ADFWI.fwi.data import (
    build_fwi_data_transform_pipeline,
    prepare_fwi_loss_pair,
    normalize_elastic_component_weights,
)
```

The package-level facade is intended for notebooks, examples, and custom
research workflows. Framework internals use owner modules such as
`ADFWI.fwi.data.preparation`, `ADFWI.fwi.data.loss`, and
`ADFWI.fwi.data.components` so implementation dependencies remain explicit.

By contrast, `ADFWI.fwi.iteration` and `ADFWI.fwi.runtime` are namespace
packages in bv1.2. Import their helpers from owner modules only, for example
`ADFWI.fwi.iteration.range` or `ADFWI.fwi.runtime.gradient`.

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
