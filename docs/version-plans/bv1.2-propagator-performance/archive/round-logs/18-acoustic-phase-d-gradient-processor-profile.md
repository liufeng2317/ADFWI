# 18 - Acoustic Phase D Gradient Processor Profile

Date: 2026-05-31

## Purpose

This round keeps the performance work focused on acoustic, but moves out of the
recurrent acoustic kernel after Phase B default-path probes stopped producing
safe improvements.

Phase D target:

```text
FWI loop / gradient-processing overhead
```

The tested hypothesis is that the existing torch-native gradient processor can
reduce CPU/NPU transfer and SciPy smoothing/preconditioning overhead for the
current Marmousi2 acoustic reduced workflow.

## Changes In This Round

No production default behavior was changed.

Two benchmark scripts were extended:

- `scripts/benchmark/gradient_processor_benchmark.py`
  - added `mask_illumination`, matching the current acoustic reduced case:
    top mute mask, forward illumination, normalization;
    JSON output now summarizes ndarray masks instead of serializing them.
- `scripts/benchmark/acoustic_fwi_iteration_profile.py`
  - added `--gradient-processor legacy|torch` so the same reduced FWI profile
    can compare legacy `GradProcessor` and `TorchGradProcessor`.

## Processor-Only Benchmark

Command:

```bash
timeout 240s conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py \
  --device npu:0 \
  --dtype float32 \
  --cases mask_illumination \
  --nx 200 \
  --nz 88 \
  --repeat 5 \
  --warmup 1 \
  --tolerance-profile npu-float32 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_gradient_processor_phase_d_benchmark_20260531.json
```

Result:

| Metric | Value |
| --- | ---: |
| legacy mean | `0.6402156815 s` |
| torch mean | `0.0057776764 s` |
| speedup | `110.81x` |
| max abs diff | `0.0222555346` |
| max rel diff | `8.9022e-06` |
| status | `ok` under `npu-float32` tolerance |

## Reduced FWI Profile

Legacy command:

```bash
timeout 600s conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 1 \
  --generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor legacy \
  --output-root /tmp/adfwi_phase_d_legacy_profile \
  --result-json /tmp/adfwi_phase_d_legacy_profile.json
```

Torch-native command:

```bash
timeout 600s conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 1 \
  --generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor torch \
  --output-root /tmp/adfwi_phase_d_torch_profile \
  --result-json /tmp/adfwi_phase_d_torch_profile.json
```

Result:

| Metric | Legacy | Torch-native | Ratio |
| --- | ---: | ---: | ---: |
| loss | `6375.7919921875` | `6375.7919921875` | exact |
| raw grad norm | `0.8779116273` | `0.8779116273` | exact |
| processed grad norm | `67952.5546875` | `67952.2890625` | rel diff `3.91e-06` |
| `vp_update_norm` | `1232.8828125` | `1232.8828125` | exact |
| gradient processing | `0.6442661881 s` | `0.0524322502 s` | `12.29x` |
| total iteration | `25.8375888076 s` | `24.1176032759 s` | `1.071x` |

## Interpretation

This is a useful acoustic-only Phase D route:

- It avoids the high-risk recurrent kernel path.
- It uses an already supported runtime dispatch path:
  `TorchGradProcessor.forward_torch`.
- The reduced FWI numerical metrics are stable:
  loss and raw gradient norm are exact; processed-gradient norm differs only at
  NPU float32 smoothing/preconditioning tolerance; the first `vp` update norm
  is exact.
- The measured total iteration improvement is moderate but real for this
  reduced profile.

## Decision

Accept torch-native gradient processing as an acoustic opt-in performance path.

Do not change the legacy default yet. The next acoustic task should validate
this path for a longer reduced inversion or the full-record validation script
before updating recommended example settings.

## Verification

Commands run:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/gradient_processor_benchmark.py \
  scripts/benchmark/acoustic_fwi_iteration_profile.py

conda run -n adfwi python -m unittest tests.test_torch_grad_processor -v

conda run -n adfwi python scripts/benchmark/gradient_processor_benchmark.py \
  --device cpu \
  --dtype float32 \
  --cases mask_illumination \
  --nx 32 \
  --nz 24 \
  --repeat 1 \
  --warmup 0 \
  --output /tmp/adfwi_gradient_processor_mask_illumination_smoke.json
```

Notes:

- `unittest` passed all 7 `TorchGradProcessor` tests.
- The CPU smoke for `mask_illumination` passed.
- `conda run -n adfwi python -m pytest ...` could not be used because the
  current `adfwi` environment does not have `pytest` installed.

## Next Direction

Stay on acoustic Phase D.

Next task:

```text
Run a short reduced acoustic inversion comparison, e.g. 10 iterations, with
legacy GradProcessor versus TorchGradProcessor. Compare loss trajectory,
final model/update metrics, runtime, and output stability.
```

If the 10-iteration comparison is stable, update the acoustic validation scripts
to expose/recommend `gradient_processor="torch"` for NPU performance while
keeping legacy as the compatibility default.
