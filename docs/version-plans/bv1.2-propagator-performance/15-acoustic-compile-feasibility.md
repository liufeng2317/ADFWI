# 15 - Acoustic Compile Feasibility Probe

Date: 2026-05-31

## Purpose

This probe checks whether `torch.compile` is a realistic Phase B optimization
route for the acoustic AD hot path on the current NPU environment.

The test is deliberately isolated from `ADFWI/propagator`. It does not modify
the production kernel. It uses a representative acoustic pressure-update
calculation and compares eager execution against compiled execution for:

- scalar loss;
- output tensor;
- gradients of `p`, `u`, and `w`;
- forward, backward, and total wall-clock time.

## Commands

Default compile backend:

```bash
timeout 240s conda run -n adfwi python scripts/benchmark/acoustic_compile_feasibility.py \
  --device npu:0 \
  --dtype float32 \
  --nx 48 \
  --nz 32 \
  --nabc 8 \
  --repeat 2 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_compile_feasibility_20260531.json
```

`backend=eager` control:

```bash
timeout 180s conda run -n adfwi python scripts/benchmark/acoustic_compile_feasibility.py \
  --device npu:0 \
  --dtype float32 \
  --nx 48 \
  --nz 32 \
  --nabc 8 \
  --repeat 2 \
  --compile-backend eager \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_compile_eager_backend_feasibility_20260531.json
```

Syntax check:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_compile_feasibility.py
```

## Environment

- backend: `npu:0`
- dtype: `float32`
- PyTorch: `2.8.0+cpu`
- torch-npu: `2.8.0.post4`
- tested modes:
  - `inner_only`
  - `sliced_assignment`

## Result

### Default `torch.compile`

The default compile backend failed before a compiled run could complete:

```text
BackendCompilerFailed: backend='inductor' raised:
ModuleNotFoundError: No module named 'triton'
```

This happened for both `inner_only` and `sliced_assignment`.

Conclusion: default `torch.compile` / inductor is not currently a usable
optimization route for the NPU acoustic propagator environment.

### `backend=eager` Control

`torch.compile(..., backend="eager")` ran successfully and preserved numerical
behavior exactly in this probe:

| Mode | loss diff | output diff | p grad diff | u grad diff | w grad diff | total speedup mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `inner_only` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `1.002x` |
| `sliced_assignment` | `0.0` | `0.0` | `0.0` | `0.0` | `0.0` | `0.953x` |

The eager backend validates that the small functions can be captured without
changing gradients, but it is not an optimizing backend. It does not justify a
production kernel integration.

## Decision

Do not continue `torch.compile` as the next production optimization task on
this branch.

Reasons:

- the default optimizing backend is blocked by the current environment
  dependency chain;
- the non-optimizing eager backend gives no useful speedup signal;
- a propagator-level compile experiment would add complexity without an
available optimizing backend.

`torch.compile` should remain a research-only option until the NPU compile
stack is explicitly prepared and can pass this same feasibility probe with an
optimizing backend.

## Next Direction

Return to the main Phase B route instead of continuing this line.

The next practical task should be a bounded default-path benchmark around
allocation/write overhead in the acoustic kernel, using output and gradient
parity as the gate. The earlier evidence still points to `slice_backward`,
`copy_`, `zero_`, and allocation pressure as the main default-path cost center.
