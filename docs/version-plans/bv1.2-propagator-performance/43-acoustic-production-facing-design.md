# Acoustic Production-Facing Custom Backward Design

Date: 2026-05-31

## Purpose

The benchmark-only custom recurrence now has a useful signal:

- outputs/loss match production exactly in reduced validation parity;
- raw `vp.grad` max abs diff is about `2.9e-7` on NPU float32;
- backward is faster (`1.34x`);
- total reduced validation iteration is slightly faster (`1.04x`);
- forward is still slower because the prototype calls a Python-level custom
  function once per timestep.

This document defines the next production-facing route. It is intentionally a
design gate before editing `ADFWI/propagator/acoustic_kernels.py`.

## Current Production Shape

Production acoustic propagation is:

```text
forward_kernel(...)
  pad model and density
  build alpha/kappa coefficients
  for each checkpoint chunk:
      step_forward(...)
```

`step_forward` is TorchScripted and owns the timestep loop. It records receiver
outputs and optionally accumulates detached forward-wavefield summaries.

This is why production forward is faster than the benchmark custom path:

- one TorchScripted `step_forward` call per chunk;
- no Python-level `torch.autograd.Function.apply` call per timestep.

## Current Benchmark Custom Shape

The benchmark path is:

```text
experimental_forward_kernel(...)
  for it in range(nt):
      CustomTimestepUpdateWithFeatures.apply(...)
      record receivers
```

This is useful for formula validation, but not a production shape. It pays:

- one Python custom autograd dispatch per timestep;
- Python list append and stack for receiver records;
- extra saved tensors per timestep.

## Production-Facing Route

### Route A: Chunk-Level Custom Autograd Wrapper

Wrap one full acoustic time chunk, not one timestep:

```text
custom_step_forward_chunk.apply(
    static config,
    src_x, src_z, src_index, src_v_chunk,
    rcv_x, rcv_z,
    kappa1, alpha1, kappa2, alpha2, kappa3,
    p0, u0, w0,
)
```

Forward:

- run the same recurrence as production `step_forward`;
- preserve returned `p`, `u`, `w`, `rcv_p`, `rcv_u`, `rcv_w`;
- preserve `forward_wavefield_*` behavior when requested;
- save only the state needed for custom backward.

Backward:

- reverse over the chunk;
- propagate receiver upstream through pressure/velocity states;
- compute gradients for `p0`, `u0`, `w0`, `kappa1`, `alpha1`, `kappa2`,
  `alpha2`, and `kappa3`;
- rely on normal autograd for coefficient dependencies back to `v` and `rho`.

Why this is the preferred route:

- one custom autograd dispatch per chunk instead of per timestep;
- keeps batched source execution;
- retains the current `forward_kernel` public contract;
- isolates the production change behind an explicit internal branch.

### Route B: Timestep-Level Production Replacement

Move `CustomTimestepUpdateWithFeatures` into production and call it inside the
existing Python timestep loop.

This route is rejected for now:

- it preserves the slow Python-level timestep dispatch;
- benchmark evidence already shows forward becomes slower;
- it creates production complexity without solving the main forward overhead.

### Route C: Adjoint-Only External Gradient Kernel

Keep production forward unchanged and compute custom gradients after the
receiver loss has produced upstreams.

This is research-only for now:

- it would bypass normal PyTorch autograd flow through `forward_kernel`;
- it needs a separate gradient injection API;
- it would change FWI internals more broadly than the propagator.

## Integration Boundary

Do not change public user-facing signatures first.

The safe internal integration point is inside `forward_kernel`:

```python
if custom_backward and checkpoint_segments == 1 and not save_forward_wavefield:
    use chunk-level custom autograd path
else:
    use current production path
```

Initial restrictions should be explicit:

- acoustic only;
- `checkpoint_segments == 1`;
- no encoded source branch until separately tested;
- `save_forward_wavefield=False` first;
- NPU/float32 validation required;
- CPU/float64 formula check required for any formula edits.

## Acceptance Gate

A production-facing prototype can be accepted only if all pass:

| Gate | Requirement |
| --- | --- |
| Direct kernel parity | outputs/loss exact; raw `v.grad` max abs around `1e-6` or better on NPU float32 |
| Reduced validation FWI parity | outputs/loss exact; finite raw gradients; raw `vp.grad` max abs around `1e-6` or better |
| Reduced iteration speed | total speedup must be meaningfully above noise, target `>=1.05x` |
| Full-record check | at least one full-record forward/backward or 10-iteration reduced/full run before default use |
| Default behavior | unchanged unless user explicitly opts in |

## Next Implementation Task

Create a benchmark-only chunk-level custom autograd prototype before touching
production:

```text
scripts/benchmark/acoustic_custom_chunk_forward.py
```

The first prototype should use a small chunk and reuse the already validated
custom backward formulas. Its purpose is to answer:

```text
Can one custom autograd dispatch per chunk preserve parity and reduce forward
overhead compared with per-timestep custom Function dispatch?
```

Only if this benchmark is positive should `ADFWI/propagator/acoustic_kernels.py`
receive a guarded production-facing implementation.
