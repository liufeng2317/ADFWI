# Deepwave Method Map

This map summarizes the design ideas observed from the local Deepwave source
and how they relate to ADFWI. It is intentionally written as a method map, not
as an implementation instruction.

## 1. Observed Deepwave Design

```text
deepwave.scalar(...)
  |
  +-- validate and normalize inputs
  +-- pad/prepare wavefields and PML profiles
  +-- call ScalarForwardFunc.apply(...)
        |
        +-- allocate intermediate storage with StorageManager
        +-- call compiled scalar forward backend
        +-- save only required tensors and storage handles
        |
        +-- backward(...)
              |
              +-- call ScalarBackwardFunc.apply(...)
                    |
                    +-- call compiled scalar backward/adjoint backend
                    +-- accumulate model/source/wavefield gradients
```

Important source references:

| Deepwave file | Relevant idea |
| --- | --- |
| `external/deepwave/src/deepwave/scalar.py` | Python API, custom autograd Function, forward/backward split |
| `external/deepwave/src/deepwave/common.py` | `StorageManager`, storage modes, temporary storage |
| `external/deepwave/src/deepwave/backend_utils.py` | C/CUDA backend loading and typed function bindings |
| `external/deepwave/src/deepwave/scalar.c` / `.cu` | compiled forward/backward kernels |
| `external/deepwave/tests/test_scalar.py` | operator-level validation style |

## 2. Ideas Worth Learning

### 2.1 Custom Autograd Boundary

Deepwave does not allow PyTorch to trace every time step as a large Python
autograd graph. It defines a custom autograd boundary:

- Python prepares tensors and metadata;
- compiled forward performs propagation;
- compiled backward performs adjoint propagation;
- PyTorch only sees a differentiable operator.

ADFWI implication:

- the most valuable next step is an explicit acoustic operator boundary;
- the existing Python autograd path should remain as the reference baseline;
- custom backward must be validated against raw `vp.grad`, not just loss.

### 2.2 Explicit Storage Policy

Deepwave separates computation from intermediate storage:

```text
storage_mode = device | cpu | disk | none
storage_compression = true | false
```

ADFWI implication:

- checkpoint count alone is not a complete memory strategy;
- storage mode should become a first-class propagator contract;
- speed-memory tradeoffs should be measured by policy, not by ad hoc flags.

### 2.3 Compiled Time Loop

Deepwave moves the time stepping loop into C/CUDA. This avoids repeated Python
dispatch, slice assignment graph creation, and large autograd graph retention.

ADFWI implication:

- Python-level micro-edits will not match this ceiling;
- if NPU custom operators are not immediately available, a CPU/CUDA prototype
  or Torch extension scaffold can still define the contract;
- the NPU route needs a dedicated backend feasibility check.

### 2.4 Narrow Public API, Rich Internal Contract

Deepwave keeps user-facing calls simple while maintaining a detailed internal
contract for shapes, storage, PML, callbacks, and gradients.

ADFWI implication:

- keep `AcousticPropagator` readable;
- expose only a small opt-in parameter for a new operator/storage policy;
- hide backend complexity behind a separate operator module.

## 3. Ideas Not To Copy Directly

| Deepwave idea | Why not direct-copy into ADFWI |
| --- | --- |
| CUDA/C backend source | ADFWI currently targets NPU as an important backend; CUDA code is not directly portable |
| Full storage mode surface immediately | Too much API complexity before the acoustic operator proves value |
| Double backward support | Useful in Deepwave, but not required for the first ADFWI FWI performance target |
| Disk storage | Likely too slow for current optimization target; keep as later memory fallback |
| Broad 1D/2D/3D/general accuracy API | ADFWI should first optimize its current acoustic contract |

## 4. ADFWI-Specific Method Graph

```text
Current ADFWI production
  |
  +-- Python/TorchScript time loop
  +-- PyTorch autograd through recurrence/checkpoint segments
  +-- pressure-only and wavefield-summary policies

Deepwave-inspired target
  |
  +-- ADFWI API compatibility
  +-- custom acoustic operator boundary
  +-- explicit adjoint gradient path
  +-- storage policy: device / remat / compressed checkpoint
  +-- backend implementation selected by device feasibility
```

## 5. Priority Ranking

| Priority | Method | Expected value | Risk |
| --- | --- | --- | --- |
| 1 | custom autograd acoustic operator contract | high | high numerical validation burden |
| 2 | explicit storage policy | high | medium API/design risk |
| 3 | compiled/extension time loop | high | backend portability risk |
| 4 | operator-level tests like Deepwave | high | low |
| 5 | Python slicing micro-optimization | low | low but mostly exhausted |

