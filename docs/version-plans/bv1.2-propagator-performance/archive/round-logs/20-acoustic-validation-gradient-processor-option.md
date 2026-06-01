# 20 - Acoustic Validation Gradient Processor Option

Date: 2026-05-31

## Purpose

Expose the validated acoustic Phase D optimization as an explicit validation
script option:

```text
--gradient-processor legacy|torch
```

This keeps the legacy NumPy/SciPy `GradProcessor` as the default compatibility
path while allowing NPU validation runs to opt into `TorchGradProcessor`.

## Code Changes

Updated:

- `examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py`
- `examples/validation/marmousi2_acoustic_full_record/scripts/inversion.py`

Changes:

- added `--gradient-processor {legacy,torch}`;
- imported `TorchGradProcessor`;
- selected processor class from the explicit CLI option;
- wrote `gradient_processor` into inversion summary JSON.

No propagator kernel behavior was changed.

## Verification

Syntax and CLI help:

```bash
conda run -n adfwi python -m py_compile \
  examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py \
  examples/validation/marmousi2_acoustic_full_record/scripts/inversion.py \
  examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py \
  examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py

conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py --help | rg -n "gradient-processor|legacy|torch"

conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/inversion.py --help | rg -n "gradient-processor|legacy|torch"
```

Reduced validation smoke:

```bash
timeout 300s conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --output-root /tmp/adfwi_validation_torch_processor_smoke

timeout 300s conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/inversion.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 1 \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor torch \
  --output-root /tmp/adfwi_validation_torch_processor_smoke
```

Smoke result:

| Field | Value |
| --- | --- |
| status | `ok` |
| gradient processor | `torch` |
| final loss | `6375.7919921875` |
| `vp_update_norm` | `1232.8828125` |
| inversion seconds | `27.2660 s` |

## Decision

Accept this as an acoustic NPU opt-in validation option.

Do not change defaults yet:

- `legacy` remains default in validation scripts;
- `torch` is now available for explicit performance runs;
- longer full-record validation should be run before using torch-native
  gradient processing as a new baseline.

## Next Direction

Stay focused on acoustic.

Recommended next step:

```text
Run a full-record short inversion, e.g. 10 iterations, with
--gradient-processor torch and compare against the existing full-record
baseline or a matching legacy 10-iteration run.
```

This will decide whether the full-record acoustic validation baseline should
recommend torch-native gradient processing for NPU performance.
