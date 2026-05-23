# bv1.2 Device Backend Interface Design

This document defines the planned `ADFWI.backends` interface before the actual
code refactor. The goal is to make device scheduling a framework-level feature,
not a repeated per-object argument.

## Design Goal

ADFWI users should be able to configure the compute device once and let the core
physical modeling workflow inherit that choice consistently.

Target user experience:

```python
from ADFWI.backends import configure_backend

configure_backend("npu:0")
```

After that, ADFWI core objects should default to the configured backend unless an
explicit `device=` override is provided.

This first phase covers ADFWI physical forward/backward workflows only. It does
not include DR-FWI, DIP, or neural-network reparameterization modules.

## Current v1.1 Problem

In v1.1, device logic is distributed across the framework:

- models store `self.device`;
- propagators store `self.device` and pass it to kernels;
- FWI classes copy `self.propagator.device`;
- regularization classes receive device directly;
- kernels receive raw `device` and `dtype` arguments;
- utilities frequently convert between tensors and CPU NumPy arrays;
- examples repeatedly pass the same `device` value to many objects.

This makes CPU/CUDA/NPU support harder to maintain because each class can drift
slightly in how it interprets and applies device settings.

## Proposed Public API

### Global Configuration

```python
from ADFWI.backends import configure_backend, get_backend

configure_backend("npu:0")
backend = get_backend()
```

`configure_backend()` sets the framework default backend. Researchers can either
request one explicit device or let ADFWI auto-select from a configurable priority
list.

Examples:

```python
# Planned local adfwi environment: prefer NPU, then CPU.
configure_backend(None, prefer=("npu", "cpu"))

# CUDA workstation: prefer CUDA, then CPU.
configure_backend(None, prefer=("cuda", "cpu"))

# Mixed accelerator machine: choose a project-specific priority.
configure_backend(None, prefer=("cuda", "npu", "cpu"))

# Fully explicit request.
configure_backend("npu:0")
```

Accepted initial device requests:

- `None`: auto-select using framework policy;
- `"cpu"`;
- `"cuda"` or `"cuda:0"`;
- `"npu"` or `"npu:0"`.

### Scoped Override

```python
from ADFWI.backends import use_backend

with use_backend("cpu"):
    # ADFWI objects created here inherit CPU by default.
    ...
```

The scoped API is useful for tests and mixed-device experiments. It should not be
required for normal examples.

### Object-Level Override

Existing explicit `device=` arguments should remain supported during migration:

```python
model = AcousticModel(..., device="cpu")
```

Migration rule:

- if `device` is explicitly provided, use it;
- if `device is None`, inherit `get_backend().device`;
- old code using `device="cpu"` should still work.

## Backend Object

The backend object should be small and stable.

Proposed fields:

```python
backend.name          # "cpu", "cuda", or "npu"
backend.device        # torch-compatible device or canonical device string
backend.index         # device index or None
backend.dtype         # default torch dtype for ADFWI tensors
backend.available     # whether requested backend is available
backend.fallback      # whether CPU fallback was used
backend.reason        # optional diagnostic message
```

Proposed methods:

```python
backend.resolve_device(device=None)
backend.to_device(x, dtype=None)
backend.tensor(data, dtype=None)
backend.zeros(shape, dtype=None)
backend.ones(shape, dtype=None)
backend.empty(shape, dtype=None)
backend.arange(*args, dtype=None)
backend.synchronize()
backend.memory_allocated()
backend.diagnostics()
```

The tensor factory helpers are intended to remove repeated low-level device and
dtype handling from models, propagators, regularization, tests, and benchmarks.

## Backend Detection Policy

Default policy for `configure_backend(None)` in the planned `adfwi` conda environment:

1. Use NPU if an NPU runtime is importable and reports availability.
2. Else use CPU.

The priority is intentionally public and configurable through `prefer`. CUDA remains supported for explicit requests or CUDA-equipped machines, but it is not the default priority for the planned CPU+NPU environment.

For explicit requests:

- requesting unavailable `cuda` or `npu` should raise a clear error by default;
- tests may opt into fallback mode explicitly;
- silent fallback should be avoided in production inversion scripts.

Possible API:

```python
configure_backend("npu:0", fallback=False)
configure_backend("npu:0", fallback=True)  # only for smoke tests or demos
```

## NPU Runtime Boundary

The backend layer should isolate optional NPU imports. The rest of ADFWI should
not import NPU-specific packages directly.

Expected pattern:

```python
# inside ADFWI.backends only
try:
    import torch_npu
except ImportError:
    torch_npu = None
```

All other modules should consume the resolved backend/device through the public
`ADFWI.backends` API.

## Migration Plan

### Phase 1: Add Backend Module Without Behavior Changes

Create:

```text
ADFWI/backends/
  __init__.py
  backend.py
```

Add tests for:

- CPU backend resolution;
- CUDA backend resolution when available;
- NPU backend resolution when available;
- unavailable backend error messages;
- scoped backend restoration.

### Phase 2: Core Constructors Inherit Backend Defaults

Change defaults conservatively:

```python
def __init__(..., device=None, dtype=torch.float32):
    backend = get_backend(device=device, dtype=dtype)
    self.device = backend.device
    self.dtype = backend.dtype
```

Initial targets:

- `AbstractModel` and concrete model classes;
- `AcousticPropagator`;
- `ElasticPropagator`;
- regularization classes.

### Phase 3: FWI Classes Use Backend Diagnostics

Update:

- `AcousticFWI`;
- `ElasticFWI`.

Keep tensor-local operations in misfits where possible. Misfits should usually
use `syn.device` or `obs.device`, not global backend state.

### Phase 4: Kernel Call Sites Stay Explicit But Backend-Fed

Do not rewrite kernels first. Keep kernel signatures such as:

```python
forward_kernel(..., device=self.device, dtype=self.dtype)
```

But ensure `self.device` and `self.dtype` come from the backend layer.

### Phase 5: Smoke Tests

Minimum tests before broader refactor:

- acoustic forward on CPU;
- acoustic backward/loss backward on CPU;
- acoustic forward on NPU target machine;
- tiny acoustic inversion with 1 iteration when accelerator memory allows.

## Implementation Status

Implemented in the current `bv1.2` working tree:

- Added `ADFWI.backends` with `configure_backend`, `get_backend`, `resolve_backend`, `use_backend`, `Backend`, and backend-specific errors.
- Default auto-selection is configurable and currently prioritizes `("npu", "cpu")` for the local `adfwi` CPU+NPU environment.
- `AbstractModel` and `AcousticModel` inherit backend `device`/`dtype` when not explicitly provided.
- `AcousticPropagator` follows the model backend by default while preserving explicit `device` override.
- Acoustic regularization classes inherit backend defaults and construct `L0`/`L1` on `self.device` with `self.dtype`.
- `AcousticFWI` validates model/propagator device consistency and aligns regularization tensors to the propagator backend during initialization.
- Added unit/integration tests for backend resolution, configurable priority, CPU/NPU inheritance, dtype inheritance, regularization inheritance, and AcousticFWI regularization alignment.

Verified with:

```bash
conda run -n adfwi python -m unittest tests/test_backends.py tests/test_backend_integration.py
```

Result on the local CPU+NPU machine: `Ran 17 tests ... OK (skipped=1)`.

Still pending:

- `ElasticPropagator` and `ElasticFWI` migration.
- Full acoustic smoke test scripts for forward, backward, and one-iteration inversion.
- Misfit/kernel audit for NPU-sensitive operations and dtype consistency.
- Public README/example update after the migration is stable.

## Acoustic Smoke Test Script

A notebook-free smoke test has been added for bv1.2 backend validation:

```bash
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device npu:0
conda run -n adfwi python scripts/smoke/acoustic_backend_smoke.py --device auto --prefer npu,cpu
```

For repeated machine checks, use the multi-device wrapper:

```bash
conda run -n adfwi python scripts/smoke/run_acoustic_backend_smoke.py --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/run_acoustic_backend_smoke.py --devices cpu,npu:0 --skip-backward
```

The wrapper invokes the single-device smoke test once per requested backend and
returns one combined JSON report. Successful child-process stderr is hidden by
default so optional runtime warnings do not obscure the baseline metrics; pass
`--include-stderr` when debugging environment warnings.

The script builds a tiny in-memory acoustic model and survey, runs one forward
pass, computes `mean(p**2)`, runs backward by default, and prints a JSON summary
with backend diagnostics, waveform shape, loss, gradient norm, and timings. It
does not read or write notebooks, figures, wavefields, or example output files.

Default smoke-test geometry:

- model: `nx=24`, `nz=20`, `nabc=4`, `dx=10 m`, `dz=10 m`;
- survey: one moment-tensor source, three pressure receivers;
- time axis: `nt=30`, `dt=0.001 s`;
- dtype: `float32`;
- backward target: `model.vp`.

Validated on the local `adfwi` CPU+NPU environment:

- CPU: status `ok`, pressure shape `[1, 30, 3]`, loss about `3.7251948e-09`, nonzero finite `vp` gradient.
- NPU `npu:0`: status `ok`, pressure shape `[1, 30, 3]`, loss about `3.7251946e-09`, nonzero finite `vp` gradient.

The CPU run may still emit Ascend owner warnings because importing the backend
checks optional NPU availability in the local environment; those warnings did not
affect the smoke-test result.

## Mini Inversion Smoke Test

A one-iteration AcousticFWI smoke test has been added to validate the full
forward/loss/backward/optimizer/model-update path without touching notebooks or
example outputs:

```bash
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0
```

The script builds an in-memory true acoustic model and a perturbed initial model,
synthesizes observed data from the true model, and runs one `AcousticFWI`
iteration on the initial model. It checks that loss, `vp` gradient norm, and
model update norm are finite and nonzero. Progress bars are hidden by default;
pass `--show-progress` when debugging the FWI loop.

Validated on the local `adfwi` CPU+NPU environment:

- CPU: status `ok`, loss about `3.803281e-07`, finite nonzero `vp` gradient, finite nonzero model update.
- NPU `npu:0`: status `ok`, loss about `3.803281e-07`, finite nonzero `vp` gradient, finite nonzero model update.

## Elastic Smoke Test Script

Elastic forward/backward backend validation has been added after the acoustic
smoke tests. The script builds a tiny in-memory isotropic elastic model and
survey, runs one forward pass, computes pressure as `-(txx + tzz)`, and runs a
scalar backward pass by default:

```bash
conda run -n adfwi python scripts/smoke/elastic_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/elastic_backend_smoke.py --device npu:0
```

The elastic backend migration currently covers:

- `IsotropicElasticModel` and `AnisotropicElasticModel` inheriting global backend defaults;
- `ElasticPropagator` following the model backend by default;
- elastic finite-difference coefficients being moved to the active backend device/dtype inside the kernel.

Validated on the local `adfwi` CPU+NPU environment:

- CPU: status `ok`, pressure shape `[1, 30, 3]`, loss about `3.293591e-02`, finite nonzero `vp` gradient.
- NPU `npu:0`: status `ok`, pressure shape `[1, 30, 3]`, loss about `3.293591e-02`, finite nonzero `vp` gradient.

## Expected User Workflow After Migration

Before:

```python
device = "cuda"
model = AcousticModel(..., device=device)
propagator = AcousticPropagator(model, survey, device=device)
regularization = regularization_Tikhonov_1order(..., device=device)
fwi = AcousticFWI(propagator, model, optimizer, scheduler, loss_fn, obs_data)
```

After:

```python
from ADFWI.backends import configure_backend

configure_backend("npu:0")

model = AcousticModel(...)
propagator = AcousticPropagator(model, survey)
regularization = regularization_Tikhonov_1order(...)
fwi = AcousticFWI(propagator, model, optimizer, scheduler, loss_fn, obs_data)
```

Explicit override remains possible:

```python
model = AcousticModel(..., device="cpu")
```

## Design Constraints

1. Preserve explicit `device=` compatibility during migration.
2. Do not add NPU imports outside `ADFWI.backends`.
3. Do not rewrite finite-difference kernels before backend tests exist.
4. Keep `.cpu().detach().numpy()` bridges explicit and documented.
5. Keep DR-FWI and DIP paths out of the first backend compatibility phase.
6. Prefer clear errors over silent fallback for production scripts.

## Open Questions

1. Should global default dtype be configured through `configure_backend`, or kept
   as per-object `dtype` for now?
2. Which NPU runtime package is standard in the `adfwi` conda environment, and what exact availability API should be used?
3. Should accelerator memory reporting be part of the backend object in phase 1, or added with benchmarks?
