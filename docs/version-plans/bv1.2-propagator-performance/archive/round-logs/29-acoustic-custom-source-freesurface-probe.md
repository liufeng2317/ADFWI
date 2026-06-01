# 29 - Acoustic Custom Source And Free-Surface Probe

Date: 2026-05-31

## Purpose

Extend the isolated custom-gradient multi-step prototype with two production
features:

- source injection after the pressure update;
- free-surface boundary writes for pressure and vertical velocity.

This still does not modify production `ADFWI/propagator/acoustic_kernels.py`.

## Script Change

Updated:

- `scripts/benchmark/acoustic_custom_multistep_update_probe.py`

Added optional flags:

- `--source-injection`
- `--free-surface-boundary-write`
- `--source-depth`
- `--source-scale`

The original no-source/no-free-surface path remains available.

## Command

```bash
timeout 480s conda run -n adfwi python \
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
  --source-injection \
  --free-surface-boundary-write \
  --source-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_multistep_source_freesurface_probe_20260531.json
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
| source injection | enabled |
| free-surface boundary write | enabled |
| receiver recording | disabled |

Across three measured pairs:

| Metric | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| output max abs diff | `0.0` |
| output max rel diff | `0.0` |
| gradient max abs diff | `1.262545623603728e-12` |
| gradient max rel diff | `0.010578576475381851` |

The largest relative gradient difference occurs on very small coefficient
gradients. The absolute gradient difference remains small.

## Timing Result

Speedup is `reference / candidate`.

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `0.8766x` | `0.8852x` | `0.8954x` |
| backward | `2.0224x` | `2.0468x` | `2.0712x` |
| total | `1.5638x` | `1.5797x` | `1.5899x` |

## Interpretation

The custom-gradient route still has a positive total speed signal after adding
source injection and free-surface boundary writes.

The forward pass is slower in this isolated Python `autograd.Function`
prototype, but backward is about `2x` faster, so the total remains faster for
the measured recurrence. This is consistent with the branch goal: reduce
backward autograd overhead.

## Decision

Accept this as another positive isolated prototype milestone.

Do not integrate into production yet.

## Next Direction

The remaining production feature before an experimental propagator path is
receiver recording:

```text
Add receiver recording to the isolated recurrence prototype and compare
recorded pressure/velocity outputs, final p/u/w, and raw gradients.
```

If receiver recording passes, the next step can be an explicitly opt-in
experimental acoustic kernel path guarded by reduced-case numerical tests.
