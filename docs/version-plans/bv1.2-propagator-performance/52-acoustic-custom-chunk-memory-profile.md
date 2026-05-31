# 52. Acoustic Custom Chunk Memory Profile

Date: 2026-05-31

## Purpose

Validate whether the segmented custom acoustic chunk path can be treated as a
checkpoint replacement, or only as an opt-in high-memory acceleration path.

This test compares one reduced Marmousi2 FWI iteration with
`checkpoint_segments=10`:

- production path: PyTorch checkpoint/rematerialization path;
- custom path: opt-in `use_custom_chunk_backward=True` segmented custom
  backward path.

Both runs use the same observed data, model size, loss, gradient processing
policy, and optimizer step.

## Commands

Production checkpoint path:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_memory_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 1 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_memory_checkpoint10_production_fullshape_20260531.json
```

Custom segmented chunk path:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_memory_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 1 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --use-custom-chunk-backward \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_memory_checkpoint10_custom_chunk_fullshape_20260531.json
```

## Result

| Path | Loss | Forward (s) | Backward (s) | Total (s) | Peak allocated (MiB) | Finite checks |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| production checkpoint | 6375.7919921875 | 4.6425 | 23.8347 | 29.2244 | 272.38 | pass |
| custom segmented chunk | 6375.7919921875 | 6.6160 | 12.7919 | 20.1476 | 7876.88 | pass |

Derived ratios:

| Metric | Value |
| --- | ---: |
| Total speedup | 1.451x |
| Backward speedup | 1.863x |
| Forward ratio | 0.702x |
| Custom / production peak allocated memory | 28.92x |

## Interpretation

The segmented custom path preserves the measured loss and finite-gradient
contract in this one-iteration run, and the speed improvement is consistent
with the previous 5-iteration validation.

However, it is not memory-equivalent to the production checkpoint path. The
custom path stores enough segment-local state to raise peak allocated memory
from `272.38 MiB` to `7876.88 MiB` on this case. Therefore, this path must be
positioned as an opt-in high-memory acceleration mode, not as a replacement for
checkpoint rematerialization when memory is constrained.

## Decision

Keep `use_custom_chunk_backward=True` opt-in. Do not make it the default and do
not describe it as checkpoint-compatible memory optimization.

The next useful acoustic performance direction is to either:

1. keep this high-memory path and make its API/documentation explicit, then
   validate it on the full-record baseline; or
2. design a true rematerializing custom chunk backward that stores only segment
   boundaries and recomputes segment internals during backward.

Given the current branch goal of measurable performance without destabilizing
AD behavior, the recommended next step is option 1: document the opt-in
contract and run a short full-record validation before considering deeper
rematerialization work.
