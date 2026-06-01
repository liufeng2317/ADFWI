# 58. Acoustic Rematerialized Div-Cache Stride Sweep

Date: 2026-06-01

## Purpose

Analyze the speed drop after removing rematerialized divergence caching and
search for a practical memory/speed compromise.

The previous two endpoints were:

```text
cache all div_*      -> 2.77x memory, 1.229x total speedup
cache no div_*       -> 1.72x memory, 1.002x total speedup
```

This round adds a controllable `divergence_cache_stride` to the experimental
rematerialized custom chunk path:

```text
0: cache no divergence terms
1: cache every divergence term
N: cache every Nth time step's divergence terms
```

The default remains `0`. This is still experimental and not wired into the
default propagator path.

## Implementation

Files:

- `ADFWI/propagator/acoustic_custom_kernels.py`
  - add `divergence_cache_stride` to
    `rematerialized_custom_chunk_forward_kernel`;
  - cache only selected `div_p/div_u/div_w` terms inside rematerialized
    backward;
  - recompute uncached divergence terms on demand.
- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`
  - add `--remat-divergence-cache-stride`.

## Validation Command Template

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode experimental-remat-chunk \
  --loss-mode observed-pressure \
  --remat-divergence-cache-stride <stride> \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_stride<stride>_fullshape_gate_20260601.json
```

## Fullshape Results

All rows are compared against production `torch.utils.checkpoint` with
`checkpoint_segments=10`.

| Remat div-cache policy | Peak memory | Memory ratio | Total speedup | Backward speedup | Forward ratio | Raw `vp.grad` max abs diff |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache none (`stride=0`) | 504.96 MiB | 1.724x | 1.002x | 1.059x | 0.772x | 2.738e-07 |
| cache every 4th (`stride=4`) | 581.45 MiB | 1.985x | 1.001x | 1.058x | 0.769x | 2.738e-07 |
| cache every 2nd (`stride=2`) | 658.28 MiB | 2.247x | 1.149x | 1.227x | 0.864x | 2.738e-07 |
| cache all (`stride=1`) | 810.92 MiB | 2.768x | 1.229x | 1.344x | 0.857x | 2.738e-07 |

Production checkpoint peak memory in these fullshape gates was `292.98 MiB`.
Receiver outputs and loss were exactly identical for all tested remat variants.

## Analysis

The speed drop is caused by recomputing divergence terms inside the reverse
sweep:

```text
cache div_*:
  rematerialization forward computes div once and stores it
  reverse step directly consumes it
  faster backward, higher peak memory

recompute div_*:
  rematerialization forward discards div
  reverse step recomputes div before manual adjoint
  lower peak memory, speed gain mostly lost
```

The tradeoff is not perfectly linear. `stride=4` reaches about `2x` memory, but
it does not recover speed. `stride=2` is the first useful middle point: it keeps
the peak memory far below full div caching while preserving a clear speedup.

## Decision

For the current Python-level rematerialized prototype:

```text
Recommended experimental point: divergence_cache_stride=2
```

Reason:

- `2.25x` peak memory may be acceptable on memory-rich NPU/GPU systems;
- `1.15x` total speedup and `1.23x` backward speedup are still measurable;
- numerical parity is unchanged relative to the other remat variants.

`stride=4` is not useful because it keeps almost no speedup while still using
about `2x` peak memory.

## Next Direction

Do not promote the remat path to default yet.

Next useful validation:

1. run short real FWI with `experimental-remat-chunk` only if the benchmark
   harness can route it through the FWI loop without public API changes; or
2. design a production-facing opt-in parameter only after full-record short FWI
   passes with `divergence_cache_stride=2`.

If full-record short FWI does not preserve this `~1.15x` gain, stop this line.
