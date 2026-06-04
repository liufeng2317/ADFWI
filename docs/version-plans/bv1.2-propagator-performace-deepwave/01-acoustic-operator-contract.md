# Acoustic Operator Contract

This document starts the implementation-level design for the Deepwave-inspired
branch. It maps the current ADFWI acoustic propagator to a future custom
operator boundary.

## 1. Current ADFWI Acoustic Boundary

Current public path:

```text
AcousticPropagator.forward(...)
  |
  +-- prepare model/survey tensors
  +-- select production kernel or pressure_remat expert path
  |
  v
forward_kernel(...)
  |
  +-- pad vp/rho into boundary region
  +-- build alpha/kappa coefficients
  +-- split source wavelet by checkpoint_segments
  +-- call step_forward or step_forward_pressure_only
  +-- return receiver records and optional illumination summaries
```

Core production files:

| File | Current responsibility |
| --- | --- |
| `ADFWI/propagator/acoustic_propagator.py` | model/survey/backend adaptation and kernel dispatch |
| `ADFWI/propagator/acoustic_kernels.py` | production Python/TorchScript finite-difference recurrence |
| `ADFWI/propagator/acoustic_custom_kernels.py` | opt-in pressure rematerialization path |

## 2. Numerical Formula Boundary

ADFWI's acoustic kernel is a staggered-grid pressure/velocity update:

```text
p(t+1) = pressure update from u(t), w(t), damp/rho/vp
source injection into p(t+1)
free-surface pressure condition
u(t+1) = horizontal velocity update from p(t+1)
w(t+1) = vertical velocity update from p(t+1)
receiver sampling from p/u/w
```

This is not identical to Deepwave's scalar operator formula. Therefore:

- Deepwave's C/CUDA code should not be copied;
- the transferable idea is the operator structure, not the formula;
- ADFWI must keep its own staggered-grid equations and boundary behavior.

## 3. Candidate Custom Operator Boundary

The first candidate should have a narrow pressure-loss contract:

```text
AcousticPressureOperator.apply(
    vp,
    rho,
    damp,
    src_x,
    src_z,
    src_v,
    rcv_x,
    rcv_z,
    metadata,
    storage_policy,
) -> rcv_p
```

Metadata should contain only non-differentiable scalar configuration:

| Name | Meaning |
| --- | --- |
| `nx`, `nz` | physical model grid size |
| `dx`, `dz` | grid spacing |
| `nt`, `dt` | time sampling |
| `nabc` | absorbing boundary width |
| `free_surface` | free-surface flag |
| `src_n`, `rcv_n` | shot/receiver counts |

Differentiable inputs for the first target:

| Tensor | Gradient required? | Reason |
| --- | --- | --- |
| `vp` | yes | FWI model parameter |
| `rho` | normally no for acoustic vp-only FWI | keep as tensor input for formula parity |
| `src_v` | no for first target | source wavelet is fixed in current validation |
| `damp` | no | boundary coefficient |

Required output for the first target:

| Output | Required? | Notes |
| --- | --- | --- |
| `rcv_p` | yes | pressure waveform used by current acoustic pressure loss |
| `rcv_u`, `rcv_w` | no | not needed in pressure-only FWI path |
| illumination summaries | no | first target must use `save_forward_wavefield=False` |
| final `p/u/w` | optional diagnostic only | not part of public first contract |

## 4. Storage Policy Boundary

The custom operator must separate computation from storage policy. This is the
main Deepwave-inspired design point.

Initial policy set:

| Policy | Meaning | First-branch role |
| --- | --- | --- |
| `device` | store needed forward states on device | speed upper bound |
| `checkpoint` | store block boundaries and replay inside backward | memory-performance candidate |
| `none` | no stored states, replay from beginning as needed | diagnostic only |

Policies not in the first implementation:

| Policy | Reason to defer |
| --- | --- |
| `cpu` | device/host transfer cost must be profiled first |
| `disk` | likely too slow for current FWI target |
| compressed storage | needs a separate numerical precision discussion |

## 5. First Implementation Boundary

First implementation should be a separate prototype path, not a production
replacement.

```text
ADFWI/propagator/acoustic_operator_*.py
  |
  +-- custom autograd Function prototype
  +-- pressure receiver output only
  +-- explicit backward/adjoint only after forward parity is confirmed
```

The existing `forward_kernel` remains the reference. The existing
`pressure_remat` path remains the current accepted opt-in path.

## 6. Validation Order

Do not start with the full 40-shot case. Use staged validation:

1. tiny synthetic forward receiver parity;
2. tiny synthetic raw `vp.grad` parity;
3. reduced Marmousi2 one-iteration parity;
4. reduced 5/10 iteration FWI trajectory;
5. full-record 40-shot 10 iteration test.

Promotion requires both numerical parity and a measured speed/memory result.

## 7. Immediate Next Implementation Task

The next concrete task is to build a small acoustic pressure-operator prototype
that only reproduces forward receiver pressure. It should not expose a new
`AcousticPropagator` option until the forward parity gate passes.

