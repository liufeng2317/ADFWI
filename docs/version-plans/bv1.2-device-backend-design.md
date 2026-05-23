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

configure_backend("cuda:0")
backend = get_backend()
```

`configure_backend()` sets the framework default backend.

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

Initial policy for `configure_backend(None)`:

1. Use CUDA if `torch.cuda.is_available()`.
2. Else use NPU if an NPU runtime is importable and reports availability.
3. Else use CPU.

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

1. Should automatic backend selection prefer CUDA before NPU, or NPU before CUDA
   on the target machine?
2. Should global default dtype be configured through `configure_backend`, or kept
   as per-object `dtype` for now?
3. Which NPU runtime package is standard on the target system, and what exact
   availability API should be used?
4. Should accelerator memory reporting be part of the backend object in phase 1,
   or added with benchmarks?
