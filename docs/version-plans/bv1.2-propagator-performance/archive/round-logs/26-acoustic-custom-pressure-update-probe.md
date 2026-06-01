# 26 - Acoustic Custom Pressure Update Probe

Date: 2026-05-31

## Purpose

Start the effective high-impact route on the current performance branch:

```text
custom-gradient / adjoint-style acoustic update work
```

This probe tests only one acoustic pressure interior update. It does not modify
the production propagator.

## Script

Added:

- `scripts/benchmark/acoustic_custom_pressure_update_probe.py`

The benchmark compares:

1. reference: normal PyTorch autograd for one pressure interior update;
2. candidate: equivalent `torch.autograd.Function` with a hand-written
   backward.

Compared quantities:

- output pressure tensor;
- scalar loss;
- gradients for `p`, `u`, `w`, `kappa`, and `alpha`;
- forward, backward, and total wall time.

## Command

```bash
timeout 240s conda run -n adfwi python \
  scripts/benchmark/acoustic_custom_pressure_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 5 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_pressure_update_probe_20260531.json
```

Syntax check:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_custom_pressure_update_probe.py
```

## Numerical Result

Across five measured pairs:

| Metric | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| output max abs diff | `0.0` |
| output max rel diff | `0.0` |
| gradient max abs diff | `2.2737367544323206e-13` |
| gradient max rel diff | `0.0001390949619235471` |

The nonzero relative gradient difference is caused by very small gradient
entries. The absolute difference is at float32 noise scale for this probe.

## Timing Result

Speedup is `reference / candidate`.

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `1.0521x` | `1.1169x` | `1.1659x` |
| backward | `1.6494x` | `1.6732x` | `1.6905x` |
| total | `1.4681x` | `1.5018x` | `1.5312x` |

## Interpretation

This is the first positive signal for the high-impact route:

- the custom backward preserves output and gradients for the isolated pressure
  update;
- it reduces backward time in the targeted update;
- it addresses the exact operator family identified by the full-record
  backward profile.

However, this is not enough for production use. The production acoustic
timestep also includes:

- source injection;
- free-surface handling;
- `u` and `w` velocity updates;
- receiver recording;
- recurrence over `nt`;
- checkpoint segmentation behavior;
- FWI loss and gradient trajectory.

## Decision

Accept this as a feasibility signal, not as a production optimization.

Continue on this branch with the next bounded step:

```text
Extend the custom-autograd prototype from pressure-only update to one complete
acoustic timestep update for p/u/w, still outside production kernel code.
```

Do not modify `ADFWI/propagator/acoustic_kernels.py` until the complete
timestep prototype passes output and raw-gradient parity.

## Next Validation Gates

The next prototype must compare:

- one-step output `p/u/w`;
- gradients for input `p/u/w` and coefficients;
- NPU timing;
- then a tiny multi-step recurrence before production integration is discussed.
