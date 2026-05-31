# Acoustic Remat State Cache Probe

Date: 2026-06-01

## Purpose

This round tested whether the rematerialized acoustic checkpoint prototype can
reduce memory by saving fewer reverse-time `p/u/w` states while preserving the
speed gained by divergence caching.

This follows the previous conclusion that partial divergence-component caching
does not preserve enough speed.

## Code Path

Changed only the experimental rematerialized custom path:

```text
ADFWI/propagator/acoustic_custom_kernels.py
scripts/benchmark/acoustic_experimental_forward_iteration_parity.py
```

The production acoustic kernel remains unchanged:

```text
ADFWI/propagator/acoustic_kernels.py
```

## Implementation

Added an experimental `state_cache_stride` option:

```text
state_cache_stride=1
  Current behavior. During rematerialized backward, replay the whole chunk and
  keep every timestep's p/u/w state before reverse accumulation.

state_cache_stride=N
  Save only every Nth p/u/w boundary state during the replay pass. During
  reverse accumulation, replay each local block from its boundary state and keep
  only the block-local states/divergence terms.
```

The intended memory effect is:

```text
global p/u/w states for chunk
  -> boundary p/u/w states + block-local p/u/w states
```

This is standard rematerialization logic, but implemented in the current Python
custom-autograd prototype.

## Validation Setup

All fullshape gates used:

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
  --remat-divergence-cache-stride 2 \
  --remat-divergence-cache-components p,u,w
```

## Numerical Result

All tested state-cache variants preserved the same parity level:

| Variant | Output max abs diff | Loss abs diff | Raw `vp.grad` max abs diff |
| --- | ---: | ---: | ---: |
| `state_cache_stride=2` | `0.0` | `0.0` | `2.738e-07` |
| `state_cache_stride=5` | `0.0` | `0.0` | `2.738e-07` |
| `state_cache_stride=20` | `0.0` | `0.0` | `2.738e-07` |

## Performance Result

Reference is the production checkpoint path with `checkpoint_segments=10`.

| Variant | Peak memory | Peak memory ratio | Total speedup | Backward speedup |
| --- | ---: | ---: | ---: | ---: |
| divergence `stride=2`, all states | `658.28 MiB` | `2.247x` | `1.149x` | `1.227x` |
| state `stride=2` | `311.53 MiB` | `1.063x` | `0.920x` | `0.934x` |
| state `stride=5` | `199.01 MiB` | `0.679x` | `0.924x` | `0.954x` |
| state `stride=20` | `166.67 MiB` | `0.569x` | `0.946x` | `0.987x` |

## Interpretation

The state-list hypothesis is confirmed: saving fewer `p/u/w` states lowers peak
memory strongly, even below the production checkpoint reference.

However, the current Python-level block replay loses the speed advantage. Even
`state_cache_stride=2` drops below production runtime. Larger strides reduce
memory further but still do not recover speed, because the extra local replay
and Python-level step orchestration dominate the saved-memory path.

Therefore this is a useful memory diagnostic and potential emergency
low-memory mode, but not the desired performance optimization.

## Decision

Do not promote `state_cache_stride>1` as the recommended acoustic performance
path.

The current best speed-preserving experimental path remains:

```text
divergence_cache_stride=2
state_cache_stride=1
```

The current best memory-reducing diagnostic path is:

```text
divergence_cache_stride=2
state_cache_stride=20
```

but it is slower than production on the fullshape gate.

## Next Direction

Stop Python-level stride enumeration.

The next effective optimization cannot be another knob on this same Python
loop. It must reduce replay overhead structurally, for example:

```text
1. Move the local block replay/backward recurrence into a lower-level fused
   implementation, or
2. Return to production PyTorch checkpoint and profile whether kernel-level
   invariant hoisting can reduce backward replay cost without custom state
   management.
```

For this branch, the recommended next step is option 2 first because it keeps
the production autograd contract intact.
