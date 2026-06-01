# Acoustic Segmented Custom Chunk Gate

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

This round extends the opt-in custom chunk path from `checkpoint_segments == 1`
to segmented execution. The intent is to validate multi-segment state passing,
receiver concatenation, and gradient continuity against the existing production
checkpoint path.

Important boundary:

```text
This is not yet a memory-equivalent replacement for PyTorch checkpoint.
```

In the custom path, `checkpoint_segments` now controls custom chunk
segmentation. It does not yet implement checkpoint rematerialization, so this
round validates numerical behavior and timing, not peak-memory equivalence.

## What Changed

Production opt-in code:

- `ADFWI/propagator/acoustic_custom_kernels.py`

Tests:

- `tests/test_backend_integration.py`

The custom path now accepts:

```python
propagator.forward(
    checkpoint_segments=10,
    save_forward_wavefield=False,
    use_custom_chunk_backward=True,
)
```

The default production path is unchanged.

## Validation Commands

Targeted tests:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/propagator/acoustic_custom_kernels.py \
  ADFWI/propagator/acoustic_propagator.py

conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_custom_chunk_backward_matches_default_receiver_loss \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_custom_chunk_backward_matches_default_receiver_loss_with_segments \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_custom_chunk_backward_rejects_unsupported_options
```

Fullshape observed-pressure gate, production checkpoint vs segmented custom
chunk:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode production-custom-chunk \
  --waveform-normalize \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --batch-size 3 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_segmented_custom_chunk_checkpoint10_fullshape_20260531.json
```

## Results

| Metric | Result |
| --- | ---: |
| Reference raw `vp.grad` finite | yes |
| Candidate raw `vp.grad` finite | yes |
| Receiver output max abs diff | `0.0` |
| Receiver output max rel diff | `0.0` |
| Loss abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.7381e-7` |
| Raw `vp.grad` max rel diff | `1.0087e-1` |
| Forward speedup | `0.737x` |
| Backward speedup | `1.925x` |
| Total speedup | `1.512x` |

Timing split:

| Component | Production checkpoint | Segmented custom chunk |
| --- | ---: | ---: |
| forward | `5.0801 s` | `6.8916 s` |
| loss evaluation | `0.0623 s` | `0.0057 s` |
| backward | `24.6393 s` | `12.7984 s` |
| total | `29.7842 s` | `19.6960 s` |

## Decision

Segmented custom chunk passes the current numerical gate for
`checkpoint_segments=10` and gives a strong one-step total speedup (`1.51x`)
because backward is nearly twice as fast. However, it is not yet a checkpoint
replacement in the memory sense because it stores per-step chunk states instead
of recomputing them during backward.

This result is useful, but the wording must stay precise:

```text
The segmented custom path is faster than PyTorch checkpoint on this tested
case, but peak-memory behavior is not yet validated and should not be assumed
to match checkpoint.
```

## Next Direction

Run a short FWI loop with `checkpoint_segments=10` for default production vs
segmented custom chunk. If the speedup transfers to multiple iterations and all
finite checks pass, decide whether to keep this as an experimental performance
mode with explicit memory caveats.

Do not make it default and do not claim checkpoint replacement until peak memory
is measured.
