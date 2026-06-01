# Acoustic Observed-Pressure Finite Chunk Gate

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

The previous reduced observed-pressure gate was invalid because production
itself produced a non-finite raw `vp.grad` in the reduced `nt=120` shape. This
round uses an already verified finite production baseline: the fullshape
reduced-validation geometry with observed-pressure loss and waveform
normalization.

## What Changed

No production code changed.

No benchmark code changed in this round. The existing
`experimental-chunk` candidate was tested on a finite observed-pressure
configuration.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --candidate-mode experimental-chunk \
  --waveform-normalize \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --batch-size 3 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_observed_finite_gate_chunk_fullshape_20260531.json
```

## Result

| Metric | Result |
| --- | ---: |
| Reference raw `vp.grad` finite | yes |
| Candidate raw `vp.grad` finite | yes |
| Receiver output max abs diff | `0.0` |
| Receiver output max rel diff | `0.0` |
| Loss abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.8871e-7` |
| Raw `vp.grad` max rel diff | `6.5558e-2` |
| Forward speedup | `0.884x` |
| Backward speedup | `1.223x` |
| Total speedup | `1.112x` |

Timing split:

| Component | Production | Chunk candidate |
| --- | ---: | ---: |
| forward | `5.8745 s` | `6.6443 s` |
| loss evaluation | `0.0379 s` | `0.0069 s` |
| backward | `16.4143 s` | `13.4262 s` |
| total | `22.3274 s` | `20.0776 s` |

## Decision

The chunk-level custom backward now has a valid observed-pressure finite gate:
outputs and loss are exact, both gradients are finite, and raw `vp.grad`
difference is consistent with the earlier validated custom-gradient level
(`~2.9e-7` max abs).

The speed signal is positive but modest at this production-interface shape:
backward improves by `1.22x`, while forward is slower. This means production
integration should not be a direct one-shot replacement yet. The next step must
focus on why chunk forward is slower and whether production integration can keep
the backward gain without adding forward overhead.

## Next Direction

Stay on Phase B/C boundary and profile the chunk candidate forward overhead
before any production kernel edit. The specific question is:

```text
Is the chunk candidate forward slower because it saves/stacks all states and
divergence tensors in Python lists, or because its forward formula is less
efficient than the scripted production step_forward?
```

If the overhead is dominated by benchmark-only state stacking, production
integration can proceed to a guarded design. If the formula path itself is
slower, the custom backward route should stay experimental until the forward
path is improved.
