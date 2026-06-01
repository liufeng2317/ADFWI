# Acoustic Custom Chunk 5-Iteration FWI Validation

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

This round checks whether the `use_custom_chunk_backward=True` single-iteration
speedup transfers to a short real FWI loop. The validation uses the same
fullshape observed-pressure setup as the finite gradient gate and compares
default production against the opt-in custom chunk path for 5 optimizer
iterations.

## What Changed

Benchmark code:

- `scripts/benchmark/acoustic_fwi_iteration_profile.py`

Added:

```text
--use-custom-chunk-backward
```

This passes `use_custom_chunk_backward=True` into `acoustic_forward_batch`.
Default behavior remains unchanged.

No propagator behavior changed in this round.

## Commands

Default production:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 5 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_5iter_default_fullshape_20260531.json
```

Opt-in custom chunk:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 5 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --use-custom-chunk-backward \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_5iter_custom_chunk_fullshape_20260531.json
```

Static check:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_fwi_iteration_profile.py
git diff --check
```

## Results

Loss trajectory:

| Iteration | Default production | Custom chunk | Abs diff |
| ---: | ---: | ---: | ---: |
| 1 | `6375.7919921875` | `6375.7919921875` | `0.0` |
| 2 | `6006.28125` | `6006.28125` | `0.0` |
| 3 | `5719.279296875` | `5719.279296875` | `0.0` |
| 4 | `5500.791015625` | `5500.791015625` | `0.0` |
| 5 | `5341.65234375` | `5341.65283203125` | `4.8828e-4` |

Finiteness:

| Check | Default | Custom chunk |
| --- | --- | --- |
| all losses finite | yes | yes |
| all raw `vp.grad` finite | yes | yes |
| all processed gradients finite | yes | yes |
| all post-step `vp` finite | yes | yes |

Timing:

| Metric | Default | Custom chunk | Speedup |
| --- | ---: | ---: | ---: |
| total 5-iteration compute | `124.3322 s` | `102.3172 s` | `1.215x` |
| mean seconds / iteration | `24.8664 s` | `20.4634 s` | `1.215x` |
| mean forward seconds / iteration | `6.7831 s` | `6.8370 s` | `0.992x` |
| mean backward seconds / iteration | `17.8973 s` | `13.4630 s` | `1.329x` |

Model update:

| Metric | Default | Custom chunk |
| --- | ---: | ---: |
| `vp_update_norm` | `4970.22314453125` | `4970.22314453125` |

## Decision

The opt-in custom chunk path transfers to a real short FWI loop for the tested
configuration. The loss trajectory is effectively unchanged, finite checks pass
for all iterations, and the measured 5-iteration compute speedup is `1.215x`.

The speedup comes from backward, not forward. This matches the original
profile-driven target.

## Remaining Boundary

This does not make the custom path the default. It still only applies to:

- acoustic propagation;
- `checkpoint_segments == 1`;
- `save_forward_wavefield=False`;
- workflows where `forw_illumination=False`.

## Next Direction

Run a longer but still bounded validation, such as 20 iterations, only if we
need stronger trajectory confidence. Otherwise the next engineering task is to
decide whether the opt-in path should be wired into an FWI-level option and
documented as an experimental performance mode.
