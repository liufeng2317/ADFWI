# Acoustic Batch-Source Custom Prototype

Date: 2026-05-31

## Purpose

The previous production-interface parity used a benchmark-only custom path that
looped over sources in Python. That was useful for numerical parity but not a
fair performance signal.

This round keeps all changes in benchmark scripts and removes that artificial
per-source loop:

```text
experimental_forward_kernel now accepts src_n > 1 and injects one source per
shot in a single batched recurrence.
```

No production propagator code was changed.

## Code Scope

Updated benchmark-only scripts:

- `scripts/benchmark/acoustic_custom_multistep_update_probe.py`
- `scripts/benchmark/acoustic_experimental_forward.py`
- `scripts/benchmark/acoustic_custom_kernel_parity_probe.py`
- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`
- `scripts/benchmark/acoustic_observed_upstream_direct_replay.py`

The custom timestep helper now supports tensor source indices:

```python
source_index = torch.arange(p_new.shape[0], device=p_new.device)
p_new[source_index, source_z, source_x] += source_value
```

## Verification

### Small Direct Kernel Parity

Command: `acoustic_custom_kernel_parity_probe.py`, `3` shots, `64x32`,
`nt=120`, `64` receivers, random linear pressure upstream.

| Metric | Value |
| --- | ---: |
| Loss abs diff | `0.0` |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `3.3881317890172014e-21` |
| Total speedup | `1.4008956422356311x` |

### Reduced Validation FWI-Style Parity

Command: `acoustic_experimental_forward_iteration_parity.py`, `3` shots,
`200x88`, `nt=3000`, `200` receivers, observed-pressure loss.

| Metric | Value |
| --- | ---: |
| Loss abs diff | `0.0` |
| Receiver output max abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.8870999813079834e-07` |
| Raw `vp.grad` max rel diff | `0.06555751711130142` |
| Forward speedup | `0.6400626684744782x` |
| Backward speedup | `1.3407201828441453x` |
| Total speedup | `1.0411403758638154x` |

Timing:

| Path | Forward | Backward |
| --- | ---: | ---: |
| Production | `6.0785 s` | `16.9257 s` |
| Batched experimental prototype | `9.4967 s` | `12.6243 s` |

### Observed-Upstream Direct Replay

| Metric | Value |
| --- | ---: |
| Loss abs diff | `0.0` |
| Receiver output max abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.8870999813079834e-07` |
| Raw `vp.grad` max rel diff | `0.06555751711130142` |

Timing:

| Path | Forward | Backward |
| --- | ---: | ---: |
| Production | `5.9937 s` | `18.0080 s` |
| Batched experimental prototype | `9.5194 s` | `14.7758 s` |

## Decision

Batch-source custom recurrence is the first benchmark-only custom path with a
positive end-to-end reduced validation signal:

- numerical outputs and loss match exactly;
- raw `vp.grad` absolute difference remains around `2.9e-7` on NPU float32;
- backward is faster (`1.34x`);
- total measured FWI-style iteration is slightly faster (`1.04x`) despite a
  slower forward path.

This is still not a production implementation. The forward pass is slower
because the prototype is still Python-level recurrence code. The useful signal
is specifically the custom backward path.

## Next Direction

Move from benchmark proof-of-concept to production-facing design:

1. Keep production `forward_kernel` outputs and public arguments unchanged.
2. Design a custom autograd wrapper around the acoustic step recurrence that
   keeps batched source execution.
3. Do not accept the production change unless the same reduced validation parity
   gate passes and the full-record or 10-iteration validation shows a practical
   speed improvement.
