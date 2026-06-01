# 21 - Acoustic Full-Record Gradient Processor 10-Iteration Check

Date: 2026-05-31

## Purpose

Validate the Phase D `TorchGradProcessor` option on the full-record Marmousi2
acoustic validation case before recommending it for real validation baselines.

This run uses the same observed data for both processors and changes only:

```text
--gradient-processor legacy
--gradient-processor torch
```

No propagator kernel code was changed in this round.

## Commands

Generate the shared observed data:

```bash
timeout 600s conda run -n adfwi python \
  examples/validation/marmousi2_acoustic_full_record/scripts/forward_modeling.py \
  --device npu:0 \
  --dtype float32 \
  --shots 40 \
  --checkpoint-segments 1 \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --output-root /tmp/adfwi_full_record_gradient_processor_10iter_shared
```

Run the legacy baseline:

```bash
timeout 1800s conda run -n adfwi python \
  examples/validation/marmousi2_acoustic_full_record/scripts/inversion.py \
  --device npu:0 \
  --dtype float32 \
  --shots 40 \
  --checkpoint-segments 1 \
  --iterations 10 \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor legacy \
  --output-root /tmp/adfwi_full_record_gradient_processor_10iter_shared
```

Run the torch-native comparison:

```bash
timeout 1800s conda run -n adfwi python \
  examples/validation/marmousi2_acoustic_full_record/scripts/inversion.py \
  --device npu:0 \
  --dtype float32 \
  --shots 40 \
  --checkpoint-segments 1 \
  --iterations 10 \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --gradient-processor torch \
  --output-root /tmp/adfwi_full_record_gradient_processor_10iter_shared
```

Detailed machine-readable result:

- `acoustic_full_record_10iter_gradient_processor_compare_20260531.json`

## Result

Shared forward observed data:

| Field | Value |
| --- | --- |
| device / dtype | `npu:0`, `float32` |
| shots / nt / receivers | `40 / 3000 / 200` |
| pressure shape | `[40, 3000, 200]` |
| forward seconds | `6.938843736425042 s` |

10-iteration inversion comparison:

| Metric | Legacy | Torch | Difference |
| --- | ---: | ---: | ---: |
| seconds | `232.00042642094195` | `240.73270207829773` | `+8.732275657355785` |
| speedup, torch over legacy | - | `0.9637262591165714x` | slower |
| initial loss | `74756.2578125` | `74756.2578125` | `0.0` |
| final loss | `57679.109375` | `57679.109375` | `0.0` |
| max loss abs diff | - | - | `0.00390625` |
| max loss rel diff | - | - | `6.634274159654736e-08` |
| `vp_update_norm` | `8813.259765625` | `8813.2587890625` | `0.0009765625` |
| `vp_update_norm` rel diff | - | - | `1.1080604974438153e-07` |

## Interpretation

The full-record numerical result is stable:

- final loss is identical;
- maximum loss relative difference is `6.63e-08`;
- model update norm relative difference is `1.11e-07`.

However, the full-record end-to-end wall time is slower with
`TorchGradProcessor` in this 10-iteration run. The earlier reduced-case result
showed clear gradient-processing savings, but the full-record workflow is
dominated by propagation/backward cost and run-to-run device scheduling enough
that the processor replacement does not improve total runtime here.

## Decision

Do not promote `TorchGradProcessor` to the default full-record acoustic
validation path.

Keep the current policy:

- `legacy` remains the default gradient processor;
- `torch` remains an explicit opt-in for NPU experiments and reduced-case
  profiling;
- future performance work should return to acoustic propagation/backward cost,
  not spend more rounds on gradient processor selection.

## Next Direction

Return to the main acoustic performance line:

```text
Profile full-record acoustic per-iteration propagation/backward cost directly,
then decide whether the next measurable target is timestep update cost,
checkpoint/rematerialization policy, or memory/output policy.
```

This avoids continuing Phase D after its full-record validation result showed
no end-to-end win.
