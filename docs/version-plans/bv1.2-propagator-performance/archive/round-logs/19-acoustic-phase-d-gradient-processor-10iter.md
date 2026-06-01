# 19 - Acoustic Phase D Gradient Processor 10-Iteration Check

Date: 2026-05-31

## Purpose

This round follows the Phase D gradient-processor profile with a longer
acoustic-only validation:

```text
legacy GradProcessor vs TorchGradProcessor over 10 reduced Marmousi2 FWI
iterations.
```

No production default behavior was changed.

## Commands

Legacy run:

```bash
timeout 1200s conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 10 \
  --generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor legacy \
  --output-root /tmp/adfwi_phase_d_10iter_shared \
  --result-json /tmp/adfwi_phase_d_10iter_legacy.json
```

Torch-native run:

```bash
timeout 1200s conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 10 \
  --no-generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor torch \
  --output-root /tmp/adfwi_phase_d_10iter_shared \
  --result-json /tmp/adfwi_phase_d_10iter_torch.json
```

Both runs use the same observed data:

```text
/tmp/adfwi_phase_d_10iter_shared/waveform/obs_data.npz
```

## Numerical Result

| Metric | Result |
| --- | ---: |
| final loss, legacy | `4776.75927734375` |
| final loss, torch | `4776.75927734375` |
| final loss abs diff | `0.0` |
| max loss abs diff over 10 iters | `0.00048828125` |
| max loss rel diff over 10 iters | `8.54e-08` |
| max raw grad norm rel diff | `1.33e-06` |
| max processed grad norm rel diff | `5.58e-06` |
| `vp_update_norm`, legacy | `8410.6181640625` |
| `vp_update_norm`, torch | `8410.6181640625` |
| all finite checks | passed |

The loss trajectory is stable. The only nonzero loss difference appears at
iteration 3 and is below single-precision practical significance for this
workflow.

## Timing Result

The timing below sums the 10 profiled iterations and excludes setup and observed
data generation.

| Component | Legacy | Torch-native | Speedup |
| --- | ---: | ---: | ---: |
| total compute | `252.5678 s` | `235.5365 s` | `1.072x` |
| forward | `66.1710 s` | `71.8259 s` | `0.921x` |
| backward | `178.9009 s` | `162.6393 s` | `1.100x` |
| gradient processing | `6.5851 s` | `0.1807 s` | `36.45x` |
| optimizer step | `0.8314 s` | `0.8125 s` | `1.023x` |

The first torch-native iteration had a slower forward timing, likely from
runtime warmup. Across 10 iterations, total compute still improved by about
`7.2%`.

## Decision

Accept `TorchGradProcessor` as a validated acoustic NPU opt-in performance
path for the reduced Marmousi2 workflow.

Do not make it the default globally yet because:

- legacy `GradProcessor` remains the compatibility path;
- torch-native smoothing/preconditioning has small float32 differences from
  the NumPy/SciPy path;
- full-record or longer inversion validation should precede default changes.

## Next Direction

Stay focused on acoustic.

Next task:

```text
Expose gradient_processor="torch" in the acoustic validation scripts as an
explicit NPU performance option, then run a short validation command to confirm
the CLI/script path uses it correctly.
```

After that, decide whether to run a full-record 10-iteration comparison before
using torch-native gradient processing for longer production baselines.
