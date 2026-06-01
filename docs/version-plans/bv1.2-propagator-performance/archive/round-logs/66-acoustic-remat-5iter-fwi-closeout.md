# Acoustic Remat 5-Iteration FWI Closeout

Date: 2026-06-01

## Purpose

This round tested whether the best current rematerialized custom-chunk
candidate transfers from one-iteration parity gates to a short real FWI loop.

The tested candidate was:

```text
candidate: experimental-remat-chunk
checkpoint_segments: 10
divergence_cache_stride: 2
divergence_cache_components: p,u,w
state_cache_stride: 1
```

No production propagator code was changed.

## Added Benchmark

```text
scripts/benchmark/acoustic_experimental_fwi_loop_compare.py
```

The benchmark runs two independent short FWI loops from the same initial model:

```text
reference: production checkpoint path
candidate: experimental rematerialized custom-chunk path
```

It records loss trajectory, gradient finite checks, timing, `vp_update_norm`,
and peak allocated memory.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --iterations 5 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --candidate-mode experimental-remat-chunk \
  --remat-divergence-cache-stride 2 \
  --remat-divergence-cache-components p,u,w \
  --remat-state-cache-stride 1 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_5iter_remat_stride2_fullshape_20260601.json
```

## Numerical Result

| Metric | Result |
| --- | ---: |
| max loss abs diff | `0.0` |
| `vp_update_norm` abs diff | `0.0` |
| reference all finite | `true` |
| candidate all finite | `true` |

Loss trajectory:

| Iteration | Production | Remat candidate | Abs diff |
| ---: | ---: | ---: | ---: |
| 1 | `6375.7919921875` | `6375.7919921875` | `0.0` |
| 2 | `6006.28125` | `6006.28125` | `0.0` |
| 3 | `5719.279296875` | `5719.279296875` | `0.0` |
| 4 | `5500.791015625` | `5500.791015625` | `0.0` |
| 5 | `5341.65234375` | `5341.65234375` | `0.0` |

The candidate preserves the short-loop inversion trajectory exactly for this
gate.

## Performance Result

| Metric | Production | Remat candidate | Ratio |
| --- | ---: | ---: | ---: |
| total 5-iteration compute | `144.8131 s` | `133.1764 s` | `1.087x` speedup |
| mean seconds / iteration | `28.9626 s` | `26.6353 s` | `1.087x` speedup |
| forward seconds | `19.7125 s` | `29.5912 s` | `0.666x` |
| backward seconds | `124.2829 s` | `103.5394 s` | `1.200x` |
| peak allocated memory | `272.52 MiB` | `617.59 MiB` | `2.266x` |

## Interpretation

The rematerialized candidate still reduces backward time, but its slower
forward pass and extra memory use leave only a modest end-to-end FWI gain.

This is materially weaker than the one-iteration remat gate:

```text
one-iteration remat stride=2: about 1.15x total speedup
5-iteration remat stride=2:  1.087x total speedup
```

The reduced gain is the important result. The candidate is scientifically safe
for this gate, but it does not justify production integration.

## Decision

Do not promote the rematerialized custom-chunk path.

Reason:

```text
It gives only about 8.7% total speedup in a real 5-iteration FWI loop while
using about 2.27x the peak memory of production checkpoint.
```

This does not meet the branch goal of improving compute efficiency while
controlling memory. It is also not a better default than the existing PyTorch
checkpoint path.

## Next Direction

Close the Python-level rematerialized custom-chunk optimization line.

Further acoustic propagator performance work should not add more remat cache
knobs. The remaining meaningful options are:

```text
1. Lower-level fused acoustic stencil prototype, outside production first.
2. Boundary-saving research path with a clearly separate adjoint contract.
3. Stop acoustic kernel performance work here and move to another measured
   bottleneck.
```

For this branch, the recommended next technical step is to write the lower-level
fused-stencil feasibility contract before implementing any new kernel code.
