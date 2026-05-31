# Acoustic Observed-Upstream Direct Replay

Date: 2026-05-31

## Purpose

This round avoids another broad scan. It tests the highest-probability branch
left by the previous records:

```text
Does the exact observed-pressure receiver upstream gradient trigger the
production/custom `vp.grad` mismatch even when replayed directly through the
kernel paths, outside the FWI loss wrapper?
```

If this direct replay failed, the next work should focus on the custom backward
formula under this upstream distribution. If it passed, the next work should
focus on FWI model/rho refresh or parameter dependency wiring.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_observed_upstream_direct_replay.py \
  --device npu:0 \
  --dtype float32 \
  --batch-size 3 \
  --checkpoint-segments 1 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_observed_upstream_direct_replay_20260531.json
```

The probe uses an isolated output directory:

```text
examples/validation/marmousi2_acoustic_reduced/outputs/observed_upstream_direct_replay
```

This avoids reading stale observed data produced by earlier benchmark shapes.

## Result

| Metric | Value |
| --- | --- |
| Receiver output max abs diff | `0.0` |
| Receiver output max rel diff | `0.0` |
| External replay loss abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `0.0003633449086919427` |
| Raw `vp.grad` max rel diff | `852.9797973632812` |
| Production raw grad norm | `0.877909779548645` |
| Experimental raw grad norm | `0.8779158592224121` |

Captured upstream distribution:

| Component | Shape | Max abs | Norm | Nonzero count |
| --- | --- | --- | --- | --- |
| `p` | `[3, 3000, 200]` | `4525.302734375` | `25987.970703125` | `1524834` |
| `u` | `[3, 3000, 200]` | `0.0` | `0.0` | `0` |
| `w` | `[3, 3000, 200]` | `0.0` | `0.0` | `0` |

Timing in this diagnostic is not an accepted speed benchmark because the
experimental path is still a Python-loop prototype:

| Path | Forward | Backward |
| --- | --- | --- |
| Production direct replay | `5.8176 s` | `18.1719 s` |
| Experimental direct replay | `28.9987 s` | `46.2204 s` |

## Decision

The direct replay failed the raw-gradient parity gate while preserving receiver
outputs and replay loss exactly.

This rules out the FWI loss wrapper and the model/rho refresh path as the
primary cause for the current mismatch. The mismatch is now localized to:

```text
experimental custom backward under the observed-pressure upstream distribution
```

The custom path must not be integrated into production.

## Next Direction

Continue within the acoustic custom-gradient route, but only at the formula
level:

1. Build a minimal observed-upstream replay case with the same upstream scale
   and sparsity pattern but smaller `nt`/grid where intermediate adjoints can be
   inspected.
2. Compare production autograd gradients against the custom backward
   recurrence for the first failing time window.
3. Do not run more wrapper/output/loss scans unless the formula-level test
   clears the mismatch.
