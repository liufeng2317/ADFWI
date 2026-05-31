# 28 - Acoustic Custom Multi-Step Update Probe

Date: 2026-05-31

## Purpose

Extend the custom-gradient route from one isolated timestep to a tiny
multi-step recurrence, still outside production `acoustic_kernels.py`.

This is the next validation gate after:

- pressure-only custom backward;
- one complete `p/u/w` timestep custom backward.

## Script

Added:

- `scripts/benchmark/acoustic_custom_multistep_update_probe.py`

The script reuses the custom one-step `p/u/w` update and repeats it for a small
number of timesteps. It compares:

- final `p/u/w` outputs;
- scalar loss;
- raw gradients for initial `p/u/w` and coefficients;
- forward, backward, and total wall time.

Still intentionally excluded:

- source injection;
- free-surface boundary writes;
- receiver recording;
- checkpoint interaction;
- production propagator integration.

## Command

The first attempt used unstable random coefficient scales and produced finite
matching outputs but numerically explosive fields. That result was discarded.
The accepted run uses smaller stable coefficient scales:

```bash
timeout 360s conda run -n adfwi python \
  scripts/benchmark/acoustic_custom_multistep_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --steps 20 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --pressure-scale 1e-3 \
  --velocity-scale 1e-6 \
  --kappa-scale 1e-3 \
  --alpha1-scale 1e-3 \
  --alpha2-scale 1e-3 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_multistep_update_probe_20260531.json
```

Syntax check:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_custom_multistep_update_probe.py
```

## Numerical Result

Case:

| Field | Value |
| --- | --- |
| device / dtype | `npu:0`, `float32` |
| steps | `20` |
| shots | `1` |
| model | `nx=100`, `nz=50`, `nabc=20` |
| source injection | disabled |
| free-surface boundary write | disabled |
| receiver recording | disabled |

Across three measured pairs:

| Metric | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| output max abs diff | `0.0` |
| output max rel diff | `0.0` |
| gradient max abs diff | `5.684341886080802e-14` |
| gradient max rel diff | `0.0015048725763335824` |

The relative gradient difference is from very small `u0/w0` gradient entries.
The absolute difference is tiny.

## Timing Result

Speedup is `reference / candidate`.

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `1.2133x` | `1.2178x` | `1.2209x` |
| backward | `1.8556x` | `1.8722x` | `1.8935x` |
| total | `1.6532x` | `1.6641x` | `1.6788x` |

## Interpretation

The custom-gradient route continues to show a real speed signal after moving
from one timestep to a 20-step recurrence:

- final outputs match exactly;
- scalar loss matches exactly;
- raw gradients match to very small absolute tolerance;
- backward speedup remains near `1.87x`.

This is a stronger result than the pressure-only and one-step probes because it
checks repeated recurrence through the custom autograd function.

## Decision

Accept this as the next positive prototype milestone.

Do not integrate into production yet.

## Next Direction

Continue with production-feature parity in the isolated prototype before any
kernel integration:

```text
Add source injection and free-surface boundary writes to the custom multi-step
prototype, then compare final outputs and raw gradients against the existing
PyTorch-autograd reference.
```

Only after that passes should we discuss an experimental propagator path.
