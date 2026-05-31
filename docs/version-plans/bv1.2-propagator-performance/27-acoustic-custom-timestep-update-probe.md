# 27 - Acoustic Custom Timestep Update Probe

Date: 2026-05-31

## Purpose

Extend the custom-gradient route from pressure-only update to one complete
acoustic `p/u/w` timestep update.

This is still an isolated benchmark and does not modify
`ADFWI/propagator/acoustic_kernels.py`.

## Script

Added:

- `scripts/benchmark/acoustic_custom_timestep_update_probe.py`

The probe compares:

1. reference: normal PyTorch autograd for one timestep update;
2. candidate: equivalent `torch.autograd.Function` with hand-written backward.

The implemented timestep includes:

- pressure update;
- horizontal particle velocity update;
- vertical particle velocity update;
- sequential dependency where `u/w` use the updated pressure field.

The prototype intentionally excludes:

- source injection;
- free-surface boundary writes;
- receiver recording;
- multi-step recurrence;
- checkpoint behavior.

Those must be added only after this isolated step passes.

## Command

```bash
timeout 240s conda run -n adfwi python \
  scripts/benchmark/acoustic_custom_timestep_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 5 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_timestep_update_probe_20260531.json
```

Syntax check:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_custom_timestep_update_probe.py
```

## Numerical Result

Across five measured pairs:

| Metric | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| output max abs diff | `0.0` |
| output max rel diff | `0.0` |
| gradient max abs diff | `1.3642420526593924e-12` |
| gradient max rel diff | `0.0012061403831467032` |

The relative gradient difference is from very small gradient entries. The
absolute difference remains near float32 numerical noise for this probe.

## Timing Result

Speedup is `reference / candidate`.

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `1.1670x` | `1.1959x` | `1.2574x` |
| backward | `1.8442x` | `1.8731x` | `1.9042x` |
| total | `1.6439x` | `1.6686x` | `1.7106x` |

## Interpretation

The custom-gradient route remains promising after including the complete
`p/u/w` update dependency for one timestep.

This result directly targets the previously measured backward bottleneck:

- fewer autograd nodes for slice assignment;
- hand-written vector-Jacobian propagation for the local update;
- output and gradients preserved in the isolated case.

It is not yet production-ready because the production kernel also has source
injection, free-surface writes, receiver recording, repeated recurrence, and
checkpoint interaction.

## Decision

Accept this as a positive prototype milestone.

Do not integrate into production yet.

## Next Direction

Continue with the next bounded gate:

```text
Extend the isolated prototype to a tiny multi-step recurrence, still outside
production acoustic_kernels.py, and compare final p/u/w outputs and raw
gradients against normal PyTorch autograd.
```

If multi-step recurrence preserves gradients and keeps a speed signal, then the
next discussion can be a guarded experimental kernel path. If it fails, stop
before touching production code.
