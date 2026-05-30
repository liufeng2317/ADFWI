# Acoustic FWI Wavefield Policy Repeat Check

Date: 2026-05-31

## Purpose

This round repeats the Phase B forward-wavefield policy comparison with
alternating order:

```text
pair 1: save_forward_wavefield=True  -> save_forward_wavefield=False
pair 2: save_forward_wavefield=False -> save_forward_wavefield=True
```

The goal is to check whether the previous single-pair opt-in speedup was stable
or mostly caused by NPU warm-state/order effects.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --compare-wavefield-policy \
  --policy-repeat 2 \
  --no-grad-forw-illumination \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_index_validation \
  --no-generate-observed \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_wavefield_policy_iteration_repeat_20260531.json
```

## Numerical Result

Across both paired runs:

| Item | Max difference |
| --- | ---: |
| loss abs diff | `0.0` |
| raw `vp.grad` max abs diff | `0.0` |
| raw `vp.grad` max rel diff | `0.0` |
| processed gradient max abs diff | `0.0` |
| processed gradient max rel diff | `0.0` |
| updated `vp` max abs diff | `0.0` |
| updated `vp` max rel diff | `0.0` |

The opt-in path is numerically identical for this reduced single-iteration FWI
case when `GradProcessor(forw_illumination=False)`.

## Timing Result

Summary across two paired runs:

| Metric | Reference `True` | Candidate `False` |
| --- | ---: | ---: |
| total mean | `25.630740489810705 s` | `22.035614031367004 s` |
| total min/max | `23.2520369309932 / 28.00944404862821 s` | `21.924157733097672 / 22.147070329636335 s` |
| forward mean | `8.63895773421973 s` | `5.906182478182018 s` |
| backward mean | `16.592788128182292 s` | `16.104719520546496 s` |

Speedup summary:

| Component | Min | Mean | Max |
| --- | ---: | ---: | ---: |
| forward | `1.1780x` | `1.4626x` | `1.7471x` |
| backward | `1.0173x` | `1.0302x` | `1.0432x` |
| total | `1.0606x` | `1.1626x` | `1.2647x` |

## Interpretation

The correctness conclusion is stable: skipping forward-wavefield summaries is
safe for this FWI path only when illumination preconditioning is disabled.

The performance conclusion is positive but bounded:

- candidate total time was stable around `22 s`;
- default/reference time varied more strongly with run order and warm state;
- the stable part of the gain is mainly forward time reduction;
- backward speedup is small, about `1.03x` on average.

Therefore this should be treated as an opt-in workflow optimization, not as the
main default-path propagator optimization.

## Decision

Accept this as a validated opt-in performance path:

```text
save_forward_wavefield=False is useful for acoustic FWI only when all active
gradient processors have forw_illumination=False.
```

Do not change the default. The existing FWI guard remains required.

## Next Direction

Return to the main Phase B line for default-path performance:

```text
Profile the default acoustic backward path at finer granularity.
```

The next diagnostic should identify whether backward time is dominated by
checkpoint rematerialization, receiver-output graph, wavefield-output graph, or
finite-difference timestep operations. No new kernel rewrite should start until
that breakdown exists.
