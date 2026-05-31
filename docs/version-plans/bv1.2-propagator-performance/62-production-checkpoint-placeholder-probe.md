# Production Checkpoint Placeholder Probe

Date: 2026-06-01

## Purpose

This probe tested a small production checkpoint optimization candidate:

```text
When save_forward_wavefield=False, avoid allocating chunk-local `(nz, nx)`
forward-wavefield zero tensors inside `step_forward`.
```

The idea was that these tensors are discarded by `forward_kernel` when
illumination is disabled, but they are still allocated during checkpoint replay.

## Candidate

The candidate temporarily changed `step_forward` so that:

```text
save_forward_wavefield=True
  returns normal `(nz, nx)` chunk-local forward-wavefield summaries

save_forward_wavefield=False
  returns empty `(0, 0)` placeholders from `step_forward`
```

The outer public `forward_kernel` record still returned `(nz, nx)` zero
wavefield tensors, so no public output shape changed.

## Validation

Fullshape reduced Marmousi2 production checkpoint gate:

```text
device: npu:0
dtype: float32
checkpoint_segments: 10
shots: 3
batch_size: 3
nx/nz/nt: 200/88/3000
loss mode: observed pressure
save_forward_wavefield: false
grad_forw_illumination: false
```

Public output shape was also checked on a tiny direct `forward_kernel` call:

```text
forward_wavefield_p/u/w: (nz, nx)
```

## Numerical Result

The production-vs-production gate stayed exact:

| Item | Difference |
| --- | ---: |
| receiver output max abs diff | `0.0` |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| raw `vp.grad` max rel diff | `0.0` |

## Timing Result

The pre-change production reference/candidate pair gave:

```text
reference total: 27.2095 s
candidate total: 26.9365 s
```

The placeholder candidate pair gave:

```text
reference total: 27.0823 s
candidate total: 26.2187 s
```

Candidate total time improved by about `2.7%` compared with the previous
candidate run, but this is below the planned acceptance threshold.

Peak memory did not improve in a meaningful way:

```text
before candidate peak: 313.65 MiB
after candidate peak: 313.25 MiB
```

## Decision

Do not accept this production code change.

Reason:

```text
The candidate preserves numerical behavior but gives less than 5% total speedup
and no meaningful memory reduction.
```

The temporary code change was reverted. The JSON benchmark records are kept so
this small-allocation candidate is not repeated.

## Next Direction

Continue production checkpoint profiling, but look for a larger replay-cost
source than chunk-local forward-wavefield placeholder allocation.

The next candidate should target repeated work inside the timestep loop or
checkpoint replay that plausibly affects backward time by more than 5%.
