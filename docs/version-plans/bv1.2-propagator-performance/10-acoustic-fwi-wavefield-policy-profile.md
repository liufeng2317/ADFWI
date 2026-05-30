# Acoustic FWI Wavefield Policy Profile

Date: 2026-05-31

## Purpose

This round continues Phase B from
`propagator-performance-master-plan.md`: acoustic AD graph and backward cost.

The diagnostic compares acoustic FWI single-iteration runtime with and without
forward-wavefield accumulation under the only safe FWI condition:

```text
GradProcessor(forw_illumination=False)
```

This is an opt-in policy diagnostic. It does not change default FWI behavior.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --compare-wavefield-policy \
  --no-grad-forw-illumination \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_index_validation \
  --no-generate-observed \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_wavefield_policy_iteration_compare_20260531.json
```

## Case

| Field | Value |
| --- | --- |
| Case | `marmousi2_acoustic_reduced` |
| Device / dtype | `npu:0`, `float32` |
| Shots | `3` |
| Receivers | `200` |
| Model | `nx=200`, `nz=88` |
| Time samples | `nt=3000` |
| `checkpoint_segments` | `1` |
| Gradient illumination | `False` |

Reference:

```text
save_forward_wavefield=True
```

Candidate:

```text
save_forward_wavefield=False
```

## Numerical Comparison

| Item | Difference |
| --- | ---: |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| raw `vp.grad` max rel diff | `0.0` |
| processed gradient max abs diff | `0.0` |
| processed gradient max rel diff | `0.0` |
| updated `vp` max abs diff | `0.0` |
| updated `vp` max rel diff | `0.0` |

Both paths produced finite loss, finite raw gradient, finite processed
gradient, and finite updated `vp`.

## Timing

Single paired run:

| Component | Reference seconds | Candidate seconds | Speedup |
| --- | ---: | ---: | ---: |
| forward | `9.852246580645442` | `6.00954738818109` | `1.6394x` |
| backward | `18.073966128751636` | `16.859430003911257` | `1.0720x` |
| total measured iteration | `28.763042218983173` | `22.894235473126173` | `1.2563x` |

Other components are small in this diagnostic. The reported total speedup is
promising, but it is a single paired run and should be repeated before using it
as a stable performance number.

## Interpretation

The result confirms the expected scientific contract:

```text
When forward illumination is disabled, detached forward-wavefield summaries
can be skipped without changing loss, raw gradients, processed gradients, or
the optimizer-updated model.
```

The performance gain comes mainly from lower forward cost, with a smaller
backward improvement. This means the existing opt-in policy is useful for
workflows that do not require illumination preconditioning.

This does not justify changing the default FWI path, because the normal
`GradProcessor` default is still `forw_illumination=True`.

## Next Direction

Continue this Phase B line for one more bounded diagnostic:

1. Repeat the policy comparison with multiple paired runs or alternating order
   to reduce NPU warm-state bias.
2. If the speedup remains meaningful, document the opt-in performance contract
   and expose it only in validation/performance scripts where
   `forw_illumination=False`.
3. If the repeated result is weak, return to the main Phase B route and profile
   backward operators at finer granularity.

Do not convert `save_forward_wavefield=False` into a global default.
