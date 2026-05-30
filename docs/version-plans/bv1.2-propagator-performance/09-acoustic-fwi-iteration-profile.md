# Acoustic FWI Iteration Profile

Date: 2026-05-31

## Purpose

This round implements Phase A from `propagator-performance-master-plan.md`:
measure one reduced differentiable acoustic FWI iteration before choosing the
next propagator optimization target.

No propagator or FWI core behavior was changed in this round. The only code
change is a benchmark harness:

```text
scripts/benchmark/acoustic_fwi_iteration_profile.py
```

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_index_validation \
  --no-generate-observed \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_fwi_iteration_profile_20260531.json
```

Observed data were reused from the existing reduced validation output:

```text
examples/validation/marmousi2_acoustic_reduced/outputs/source_index_validation/waveform/obs_data.npz
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
| `save_forward_wavefield` | `True` |

## Result

Single measured iteration:

| Component | Seconds | Fraction |
| --- | ---: | ---: |
| backward | `17.784051010385156` | `61.01%` |
| forward | `9.90279239229858` | `33.97%` |
| optimizer step | `0.7517095599323511` | `2.58%` |
| gradient processing | `0.6703059785068035` | `2.30%` |
| loss evaluation | `0.03760790079832077` | `0.13%` |
| zero grad | `0.001199830323457718` | `<0.01%` |
| wavefield accumulation | `0.0002578124403953552` | `<0.01%` |
| total measured iteration | `29.147924484685063` | `100%` |

Numerical sanity:

| Metric | Value |
| --- | --- |
| loss | `6375.7919921875` |
| loss finite | `True` |
| processed `vp` grad norm | `67952.5546875` |
| processed grad finite | `True` |
| updated `vp` finite | `True` |
| `vp_update_norm` | `1232.8828125` |

The loss matches the first reduced-validation inversion loss previously used
as a bv1.2 smoke result, so the harness is measuring the same effective
iteration path.

## Interpretation

The dominant measured cost is automatic differentiation backward:

```text
backward 61.01% > forward 33.97% >> optimizer/gradient processing/loss
```

This means the next optimization should not focus on plotting, loss evaluation,
gradient post-processing, or wrapper cleanup. Those components are below the
current meaningful threshold for this case.

The next target should stay on the Phase B line:

```text
Acoustic AD graph and backward cost
```

## Next Direction

Continue the main route with a bounded Phase B diagnostic:

1. Compare default acoustic FWI iteration timing with
   `save_forward_wavefield=False` under `forw_illumination=False`.
2. Compare loss, raw `vp.grad`, processed gradient, and runtime.
3. Decide whether forward-wavefield graph/output pruning is a meaningful
   default-safe or opt-in performance path.

If this diagnostic does not show meaningful backward or total-iteration impact,
return to Phase B and profile the backward graph at a finer operator level
before editing kernel code.
