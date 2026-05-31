# 53. Acoustic Custom Chunk Full-Record Validation

Date: 2026-05-31

## Purpose

Validate whether the opt-in acoustic custom chunk path can transfer from the
reduced benchmark to a full-record Marmousi2 FWI workload.

This record does not change production propagator logic. It uses the current
opt-in path:

```text
use_custom_chunk_backward=True
checkpoint_segments=10
save_forward_wavefield=False
grad_forw_illumination=False
```

## Full-Batch Result

The first full-record attempt used all `40` shots in one batch:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case full_record \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 5 \
  --shots 40 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --use-custom-chunk-backward \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_checkpoint10_custom_chunk_5iter_20260531.json
```

Result: failed with NPU out-of-memory during custom chunk forward state
stacking.

The error occurred with roughly:

```text
60.96 GiB total capacity
51.51 GiB already allocated
1.39 GiB free
requested allocation: 1.71 GiB
```

This confirms the previous memory-profile conclusion: the custom path is not a
checkpoint memory replacement.

## Batch-20 Three-Iteration Validation

To keep the same full-record data while reducing per-batch memory, the next
test used `40` total shots with `batch_size=20`.

Production command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case full_record \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 3 \
  --shots 40 \
  --batch-size 20 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_checkpoint10_production_batch20_3iter_20260531.json
```

Custom command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case full_record \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --iterations 3 \
  --shots 40 \
  --batch-size 20 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --use-custom-chunk-backward \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_checkpoint10_custom_chunk_batch20_3iter_20260531.json
```

| Metric | Production | Custom chunk | Ratio |
| --- | ---: | ---: | ---: |
| Mean total / iter | 55.0011 s | 37.5882 s | 1.463x |
| Mean forward / iter | 7.8094 s | 12.4338 s | 0.628x |
| Mean backward / iter | 46.9299 s | 24.8831 s | 1.886x |
| `vp_update_norm` | 3395.364990 | 3395.364746 | abs diff 2.44e-4 |

Loss trajectory:

| Iteration | Production | Custom chunk | Abs diff |
| --- | ---: | ---: | ---: |
| 1 | 74756.26171875 | 74756.26171875 | 0.0 |
| 2 | 71691.45703125 | 71691.45703125 | 0.0 |
| 3 | 68890.9609375 | 68890.9609375 | 0.0 |

All finite checks passed for raw gradients, processed gradients, loss, and
post-optimizer `vp`.

## Single-Iteration Tensor Parity

The direct one-iteration parity command compared receiver outputs, loss, and
raw `vp.grad`:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case full_record \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 40 \
  --batch-size 20 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode production-custom-chunk \
  --loss-mode observed-pressure \
  --output-root examples/validation/marmousi2_acoustic_full_record/outputs/phase_a_profile \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_checkpoint10_custom_chunk_batch20_1iter_parity_20260531.json
```

| Metric | Value |
| --- | ---: |
| Receiver output max abs diff | 0.0 |
| Receiver output max rel diff | 0.0 |
| Loss abs diff | 0.0 |
| Raw `vp.grad` max abs diff | 4.470348358154297e-07 |
| Raw `vp.grad` max rel diff | 0.0018026138423010707 |
| Forward speed ratio | 0.649x |
| Backward speedup | 1.914x |
| Total speedup | 1.486x |

## Decision

The full-record validation supports the custom chunk path as an opt-in
high-memory acceleration mode when batch size is controlled.

It does not support enabling the path by default:

- full-batch `40` shots OOMs on the tested NPU;
- batch-size control is required for full-record use;
- the path is faster because backward is faster, while forward is slower;
- it remains a speed-vs-memory tradeoff rather than a checkpoint-compatible
  rematerialization strategy.

## Next Direction

Stop expanding this path as a default optimization.

The next useful task is to formalize the API contract:

```text
use_custom_chunk_backward=True is an expert opt-in speed mode for acoustic FWI.
It requires save_forward_wavefield=False and enough device memory. Users should
reduce batch_size when full-batch execution OOMs.
```

After that, choose one of two routes:

1. keep it as a documented opt-in feature and move to another measured
   bottleneck;
2. start a separate high-risk design for true rematerializing custom backward.
