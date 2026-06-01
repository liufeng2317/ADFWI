# Acoustic Remat Divergence Component Cache

Date: 2026-06-01

## Purpose

This round tested whether the rematerialized acoustic checkpoint prototype can
keep most of the full `div_p/div_u/div_w` cache speedup while storing fewer
divergence tensors.

The goal is memory reduction after a speed-improved path has been identified,
not a broad speed/memory balance search.

## Code Path

Changed only the experimental custom-autograd path:

```text
ADFWI/propagator/acoustic_custom_kernels.py
scripts/benchmark/acoustic_experimental_forward_iteration_parity.py
```

The production acoustic kernel remains unchanged:

```text
ADFWI/propagator/acoustic_kernels.py
```

## Implementation

Added an experimental `divergence_cache_components` option for the
rematerialized custom chunk path:

```text
p
u
w
p,u
p,u,w
none
```

The first implementation showed that partial caching still recomputed all three
divergence fields when any one component was missing. That made single-component
caching both memory-expensive and slow.

The implementation was then tightened so reverse recomputation is component
aware:

- missing `div_p` recomputes only `div_p`;
- missing `div_u` or `div_w` rebuilds `p_new` once and recomputes only the
  missing velocity divergence component;
- cached components are passed directly to the manual backward step.

## Validation Command Shape

All gates used the reduced Marmousi2 fullshape observed-pressure benchmark:

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
  --loss-mode observed-pressure
```

## Numerical Result

All tested component-cache variants preserved the expected parity level:

| Variant | Output max abs diff | Loss abs diff | Raw `vp.grad` max abs diff |
| --- | ---: | ---: | ---: |
| `p` before split recompute | `0.0` | `0.0` | `2.738e-07` |
| `u` before split recompute | `0.0` | `0.0` | `2.738e-07` |
| `w` before split recompute | `0.0` | `0.0` | `2.738e-07` |
| `p` after split recompute | `0.0` | `0.0` | `2.738e-07` |
| `p,u` after split recompute | `0.0` | `0.0` | `2.738e-07` |

The gradient relative error remains large in the printed max-relative metric
because the denominator includes near-zero gradient entries. The absolute error
is the relevant parity guard used in the previous rematerialized gates.

## Performance Result

Reference is the production checkpoint path with `checkpoint_segments=10`.

| Variant | Peak memory ratio | Total speedup | Backward speedup | Decision |
| --- | ---: | ---: | ---: | --- |
| no div cache | `1.72x` | `1.002x` | `1.059x` | memory lower, speed gone |
| `p` before split recompute | `2.07x` | `1.030x` | `1.102x` | not enough speed |
| `u` before split recompute | `2.07x` | `1.042x` | `1.102x` | not enough speed |
| `w` before split recompute | `2.07x` | `1.003x` | `1.056x` | not useful |
| `p` after split recompute | `2.07x` | `1.072x` | `1.138x` | improved, still weak |
| `p,u` after split recompute | `2.42x` | `1.039x` | `1.109x` | worse than expected |
| stride `2`, all components | `2.25x` | `1.149x` | `1.227x` | best current memory-focused point |
| full all-component cache | `2.77x` | `1.229x` | `1.344x` | fastest, too much memory |

## Interpretation

Partial component caching does not preserve the full-cache speedup on this NPU
benchmark. The full-cache speed gain mainly comes from avoiding the complete
reverse-time divergence recomputation workload. Storing only one or two
components still leaves enough recomputation and Python/autograd overhead that
the total speedup drops below a useful threshold.

The component-aware recompute implementation is useful diagnostically, but it
does not provide a better candidate than the existing all-component
`divergence_cache_stride=2` experiment.

## Decision

Do not promote component-level divergence caching as the main optimization.

Keep it as an experimental diagnostic knob only. The current best Python-level
rematerialized checkpoint candidate remains:

```text
divergence_cache_stride=2
divergence_cache_components=p,u,w
```

## Next Direction

Continue memory reduction from the current speed-improved rematerialized path,
but do not spend more time on partial divergence component caching.

The next meaningful direction is to reduce the memory footprint of stored
reverse states themselves, not just divergence tensors. The specific next task:

```text
Measure p/u/w state-list memory contribution inside rematerialized backward,
then test whether storing only chunk boundary states plus local reverse replay
can preserve the stride=2 speed path with lower peak memory.
```

This is the main line because state lists are now the remaining large memory
source after divergence cache reduction.
