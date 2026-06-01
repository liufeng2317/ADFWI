# 22 - Acoustic Full-Record Iteration Profile

Date: 2026-05-31

## Purpose

Return to the main acoustic performance line after Phase D showed no
full-record end-to-end win from changing the gradient processor.

This round profiles one full-record Marmousi2 acoustic FWI iteration and does
not change propagator or FWI numerical behavior.

## Script Update

Updated:

- `scripts/benchmark/acoustic_fwi_iteration_profile.py`

Change:

- added `--validation-case reduced|full_record`;
- default remains `reduced`;
- existing reduced profiling behavior is preserved;
- full-record uses the existing
  `examples/validation/marmousi2_acoustic_full_record/scripts/forward_modeling.py`
  case definition.

This is measurement tooling only.

## Command

The observed data came from the previous full-record validation output root:

```bash
timeout 600s conda run -n adfwi python \
  scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case full_record \
  --device npu:0 \
  --dtype float32 \
  --shots 40 \
  --checkpoint-segments 1 \
  --iterations 1 \
  --no-generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --output-root /tmp/adfwi_full_record_gradient_processor_10iter_shared \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_iteration_profile_20260531.json
```

Verification:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_fwi_iteration_profile.py
```

## Result

Case:

| Field | Value |
| --- | --- |
| case | `marmousi2_acoustic_full_record` |
| device / dtype | `npu:0`, `float32` |
| shots / receivers / nt | `40 / 200 / 3000` |
| model size | `200 x 88` |
| `checkpoint_segments` | `1` |
| `save_forward_wavefield` | `true` |
| `grad_forw_illumination` | `true` |
| gradient processor | `legacy` |
| loss | `74756.2578125` |
| raw `vp.grad` norm | `7.540781021118164` |
| processed grad norm | `73859.984375` |
| `vp_update_norm` | `1232.8828125` |

Timing:

| Component | Seconds | Fraction |
| --- | ---: | ---: |
| backward | `16.926167335361242` | `59.86%` |
| forward | `9.832977870479226` | `34.78%` |
| optimizer step | `0.7822253610938787` | `2.77%` |
| gradient processing | `0.6908783782273531` | `2.44%` |
| loss evaluation | `0.040355538949370384` | `0.14%` |
| zero grad | `0.001620173454284668` | `0.01%` |
| wavefield accumulation | `0.00029418617486953735` | `<0.01%` |
| total measured iteration | `28.274518843740225` | `100%` |

Setup took `56.924381762742996 s`, mostly outside the per-iteration compute
target. It should not be mixed with kernel/runtime conclusions.

## Interpretation

The full-record profile matches the earlier reduced-case conclusion:

- acoustic backward dominates the iteration;
- forward propagation is the second largest cost;
- gradient processing is not a current full-record bottleneck;
- loss construction and wavefield accumulation are negligible in this profile.

The next useful optimization must target the differentiable propagation path,
not notebook/script overhead or gradient post-processing.

## Decision

No production code optimization is accepted in this round. This is a profiling
gate.

The next acoustic task should stay on the main line:

```text
Run a full-record acoustic backward operator profile or a bounded checkpoint /
rematerialization diagnostic that explains the backward cost before changing
the kernel.
```

Do not continue Phase D unless a future profile shows gradient processing has
become dominant.
