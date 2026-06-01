# Fused Stencil Gate 0 Implementation Path

Date: 2026-06-01

## Purpose

This round executed Gate 0 from
`67-fused-acoustic-stencil-feasibility.md`: check whether the current `adfwi`
environment has a practical lower-level fused operator path before writing any
new acoustic stencil prototype.

No production propagator code was changed.

## Environment Observed

| Item | Result |
| --- | --- |
| Python | `3.9.25` |
| PyTorch | `2.8.0+cpu` |
| NPU runtime | available through `torch_npu` |
| NPU devices | `8` |
| CUDA | unavailable |
| `torch.compile` | present |
| Ascend toolkit | present |
| `msopgen` | `/usr/local/Ascend/ascend-toolkit/latest/bin/msopgen` |
| C/C++ compiler | `/usr/bin/gcc`, `/usr/bin/g++`, `/usr/bin/c++` |
| `torch_npu.utils.cpp_extension.NpuExtension` | present |

The NPU execution stack itself is valid:

```text
torch.npu.is_available(): True
simple NPU autograd: pass
```

## Path Checks

### 1. `torch.compile`

Command shape:

```bash
conda run -n adfwi python -c '... torch.compile(f) on npu:0 ...'
```

Result:

```text
failed
ModuleNotFoundError: No module named 'triton'
```

The failure comes from the NPU inductor path importing
`torch_npu._inductor`, which requires `triton`.

Decision:

```text
Do not use torch.compile as the fused stencil route in this environment.
```

The `backend="eager"` mode works on a tiny NPU function, but it is not a fused
kernel implementation and should not be treated as a performance path.

### 2. Generic PyTorch C++ Inline Extension

Command shape:

```bash
timeout 120s conda run -n adfwi python -c 'load_inline(add_one extension)'
```

Result:

```text
did not produce a stable usable result within the timeout window
left a defunct ninja child process
```

Decision:

```text
Do not use generic torch.utils.cpp_extension.load_inline as the immediate
fused stencil route.
```

This does not prove C++ extension support is impossible, but it is not a
low-friction path for the next acoustic stencil prototype.

### 3. Ascend Custom Operator Path

Observed:

```text
msopgen exists
torch_npu.utils.cpp_extension.NpuExtension exists
Ascend toolkit and runtime library paths are configured
```

Decision:

```text
This is the only plausible lower-level fused operator route found by Gate 0.
```

However, it is not a small Python benchmark edit. It is an Ascend/CANN custom
operator project with separate build, registration, kernel implementation, and
autograd binding requirements.

## Gate 0 Decision

The branch should not proceed by rewriting more Python/TorchScript stencil
expressions.

The feasible choices are now:

```text
1. Open a focused Ascend custom operator prototype for one acoustic stencil
   update, outside production.
2. Stop acoustic kernel performance work here.
3. Treat boundary-saving as a separate research task with its own adjoint
   contract.
```

The recommended next step is option 1 only if we are ready to spend effort on a
real Ascend custom-op prototype. Otherwise, stop this acoustic performance line.

## If Continuing: Minimal Ascend Prototype Scope

The first custom-op prototype should not implement the whole propagator.

Recommended first target:

```text
fused_pressure_update_forward
```

Inputs:

```text
p, u, w, kappa1, alpha1, free_surface_start
```

Output:

```text
p_next
```

It should initially test forward-only parity against the production pressure
slice update. Autograd should not be claimed until a backward implementation is
explicitly added and validated.

Required first gate:

```text
CPU/reference production pressure update vs NPU custom-op output
max abs diff
max rel diff
shape/dtype/device checks
forward timing
```

Only after that passes should a custom backward or full p/u/w recurrence be
considered.

## Stop Condition For Next Round

Stop immediately if:

```text
1. Ascend custom-op scaffolding cannot build in the shared repo environment;
2. the build requires generated files or install steps that are too intrusive
   for the ADFWI package;
3. forward-only pressure update parity fails;
4. the first custom op is slower than the production pressure update
   microbenchmark.
```
