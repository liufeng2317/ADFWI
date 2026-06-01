# Acoustic Segmented Custom Chunk 5-Iteration FWI Validation

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

This round checks whether the `checkpoint_segments=10` segmented custom chunk
single-step gain transfers to a short real FWI loop. The comparison uses the
same fullshape observed-pressure setup as the previous gates.

## What Changed

No code changed in this round.

This is a validation-only record for:

```text
production checkpoint_segments=10
vs
custom chunk checkpoint_segments=10
```

## Commands

Default production checkpoint:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 5 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_5iter_checkpoint10_default_fullshape_20260531.json
```

Segmented custom chunk:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 5 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --use-custom-chunk-backward \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_5iter_checkpoint10_custom_chunk_fullshape_20260531.json
```

## Results

Loss trajectory:

| Iteration | Production checkpoint | Segmented custom chunk | Abs diff |
| ---: | ---: | ---: | ---: |
| 1 | `6375.7919921875` | `6375.7919921875` | `0.0` |
| 2 | `6006.28125` | `6006.28125` | `0.0` |
| 3 | `5719.279296875` | `5719.279296875` | `0.0` |
| 4 | `5500.791015625` | `5500.791015625` | `0.0` |
| 5 | `5341.65234375` | `5341.65234375` | `0.0` |

Finiteness:

| Check | Production checkpoint | Segmented custom chunk |
| --- | --- | --- |
| all losses finite | yes | yes |
| all raw `vp.grad` finite | yes | yes |
| all processed gradients finite | yes | yes |
| all post-step `vp` finite | yes | yes |

Timing:

| Metric | Production checkpoint | Segmented custom chunk | Speedup |
| --- | ---: | ---: | ---: |
| total 5-iteration compute | `131.8334 s` | `91.0054 s` | `1.449x` |
| mean seconds / iteration | `26.3667 s` | `18.2011 s` | `1.449x` |
| mean forward seconds / iteration | `3.8583 s` | `5.9218 s` | `0.652x` |
| mean backward seconds / iteration | `22.3510 s` | `12.1229 s` | `1.844x` |

Model update:

| Metric | Production checkpoint | Segmented custom chunk |
| --- | ---: | ---: |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |

## Decision

The segmented custom chunk path transfers to a short real FWI loop for the
tested `checkpoint_segments=10` configuration. Loss trajectories match exactly,
all finite checks pass, and total compute improves by `1.45x`.

The speedup again comes from backward. Forward is slower, but checkpoint
backward overhead is much larger, so the net loop speed is clearly positive.

## Remaining Boundary

This still does not prove checkpoint memory equivalence. The custom path uses
segmented custom autograd, but it stores per-step states inside each segment
rather than using PyTorch checkpoint rematerialization.

## Next Direction

Before documenting this as an experimental performance mode, measure peak
device memory for:

- production `checkpoint_segments=10`;
- segmented custom chunk `checkpoint_segments=10`;
- production/custom `checkpoint_segments=1` if useful as an upper-bound
  reference.

Only after memory is measured should this be positioned relative to checkpoint.
