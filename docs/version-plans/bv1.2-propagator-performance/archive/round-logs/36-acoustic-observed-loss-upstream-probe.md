# Acoustic Observed-Loss Upstream Probe

Date: 2026-05-31

## Purpose

The receiver-output locator showed exact receiver parity after matching the
production wrapper contract, but raw `vp.grad` still differed by `3.63e-4`.
This record separates the observed-pressure loss from the propagator backward.

The two questions are:

1. Does observed-pressure loss produce the same upstream gradient with respect
   to receiver records?
2. If the same upstream gradient is attached directly to receiver records, do
   production autograd and the custom backward produce the same `vp.grad`?

## Probe

Added:

- `scripts/benchmark/acoustic_observed_loss_upstream_probe.py`

The script runs four branches:

- production forward + observed-pressure loss;
- experimental forward + observed-pressure loss;
- production forward + external receiver upstream from the production loss;
- experimental forward + the same external receiver upstream.

## Command

```bash
timeout 4800s conda run -n adfwi python scripts/benchmark/acoustic_observed_loss_upstream_probe.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --checkpoint-segments 1 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/observed_loss_upstream_probe \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_observed_loss_upstream_probe_20260531.json
```

## Result

Reference record:

- `acoustic_observed_loss_upstream_probe_20260531.json`

| Metric | Value |
| --- | --- |
| Observed loss diff | `0.0` |
| Observed receiver output max abs diff | `0.0` |
| Observed receiver upstream max abs diff | `0.0` |
| Observed raw `vp.grad` max abs diff | `0.0003633449086919427` |
| Observed raw `vp.grad` max rel diff | `852.9797973632812` |
| External-upstream loss diff | `0.0` |
| External-upstream output max abs diff | `0.0` |
| External-upstream raw `vp.grad` max abs diff | `0.0003633449086919427` |
| External-upstream raw `vp.grad` max rel diff | `852.9797973632812` |

## Interpretation

Observed-pressure loss is not the source of the mismatch:

- receiver outputs match exactly;
- the upstream gradient with respect to receiver records matches exactly;
- applying the exact same upstream gradient directly to both paths preserves
  the same raw `vp.grad` mismatch.

Therefore the remaining issue is in the custom backward response to an
observed-loss-shaped upstream gradient.

This also explains the earlier synthetic-energy pass: the custom backward can
match production for simple smooth upstream gradients, but not for the upstream
gradient produced by observed-pressure residual loss.

## Decision

Do not integrate the experimental path.

The active target is now custom backward localization under a fixed external
upstream gradient.

## Next Direction

```text
Feed controlled upstream gradients directly into receiver records and vary the
component support: pressure-only, u-only, w-only, sparse time windows, and
single time/receiver impulses. Compare raw `vp.grad` to locate whether the
pressure update, source/free-surface boundary, or velocity update adjoint terms
are responsible.
```
