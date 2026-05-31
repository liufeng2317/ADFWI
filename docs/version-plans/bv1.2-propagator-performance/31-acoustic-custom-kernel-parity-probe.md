# Acoustic Custom Kernel Parity Probe

Date: 2026-05-31

## Purpose

This is the first parity gate between the production acoustic kernel interface
and the experimental custom-gradient recurrence.

Previous probes compared custom-gradient recurrence against an equivalent
Python recurrence. This probe instead compares:

- production `ADFWI.propagator.acoustic_kernels.forward_kernel`;
- experimental custom-gradient recurrence with the same tiny model, source,
  receiver, and coefficient construction.

No production kernel code is changed.

## Change

Added:

- `scripts/benchmark/acoustic_custom_kernel_parity_probe.py`

The script builds a tiny acoustic case, runs both paths, and compares:

- receiver `p/u/w` records;
- scalar receiver-record loss;
- raw velocity-model gradient `v.grad`;
- forward, backward, and total wall time.

The custom path computes the same padded `v/rho`, `alpha/kappa` coefficients,
source injection, free-surface boundary write, and receiver recording used by
the production path.

## Command

```bash
timeout 480s conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --nx 32 \
  --nz 24 \
  --nabc 8 \
  --nt 20 \
  --receivers 16 \
  --source-depth 1 \
  --receiver-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_probe_20260531.json
```

## Result

Reference record:

- `acoustic_custom_kernel_parity_probe_20260531.json`

| Metric | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Model | `nx=32`, `nz=24`, `nabc=8` |
| Time steps | `20` |
| Receivers | `16` |
| Loss absolute difference | `0.0` |
| Output maximum absolute difference | `0.0` |
| Output maximum relative difference | `0.0` |
| `v.grad` maximum absolute difference | `4.1359030627651384e-25` |
| `v.grad` maximum relative difference | `4.1359030458244794e-13` |
| Forward mean speedup | `1.2183114849102419x` |
| Backward mean speedup | `1.970601422677768x` |
| Total mean speedup | `1.6723267968751807x` |

## Interpretation

This probe passes the first production-interface parity gate:

- public receiver outputs match exactly;
- loss matches exactly;
- raw velocity gradient matches to numerical noise;
- the backward speedup signal remains close to `2x` on the tiny NPU case.

The timing is not yet a production performance claim. The custom path is still
an experimental Python harness, not an integrated kernel implementation. The
result only justifies continuing toward a production-facing opt-in prototype.

## Decision

Continue this route. The next task can move from parity harness to an explicit
opt-in experimental acoustic kernel path, but it must remain disabled by
default.

## Next Direction

```text
Implement an opt-in experimental acoustic forward path that is callable from a
benchmark script, not from normal user examples. Compare it against production
`forward_kernel` on a tiny differentiable case first, then on the reduced
Marmousi2 iteration only if tiny-case receiver outputs, loss, and raw `vp.grad`
match.
```
