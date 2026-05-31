# Acoustic Receiver Difference Location

Date: 2026-05-31

## Purpose

The previous localization record showed a `6.52e-09` receiver-output absolute
difference under the reduced Marmousi2 FWI-style benchmark. This record checks
where that difference first appears by shot, component, time index, and
receiver index.

## Locator

Added:

- `scripts/benchmark/acoustic_receiver_difference_locator.py`

The locator compares production `AcousticPropagator.forward` against the
benchmark-only experimental forward path and reports, for each component:

- first nonzero receiver difference;
- maximum difference location;
- per-shot maximum difference;
- time indices with largest accumulated difference.

## Command

```bash
timeout 2400s conda run -n adfwi python scripts/benchmark/acoustic_receiver_difference_locator.py \
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
  --threshold 0.0 \
  --top-k 12 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/receiver_difference_locator_fullsize \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_receiver_difference_locator_20260531.json
```

## Receiver Difference Result

Reference record:

- `acoustic_receiver_difference_locator_20260531.json`

| Component | First difference | First nonzero time | Max abs diff | Max rel diff | Nonzero count |
| --- | --- | --- | --- | --- | --- |
| `p` | `null` | `null` | `0.0` | `0.0` | `0` |
| `u` | `null` | `null` | `0.0` | `0.0` | `0` |
| `w` | `null` | `null` | `0.0` | `0.0` | `0` |

There is no shot/component/time/receiver location to report after the
production model-refresh contract is matched.

## Cause Of The Previous `6.52e-09` Difference

The locator exposed a benchmark-contract mismatch:

- production `AcousticPropagator.forward()` calls `model.forward()` before
  propagation;
- the experimental benchmark path initially called `experimental_forward_kernel`
  directly and skipped that model refresh.

After adding the same model refresh to the benchmark-only experimental batch,
the reduced Marmousi2 receiver outputs match exactly:

Reference record:

- `acoustic_experimental_forward_iteration_parity_refreshed_20260531.json`

| Metric | Value |
| --- | --- |
| Loss abs diff | `0.0` |
| Output max abs diff | `0.0` |
| Output max rel diff | `0.0` |
| Raw `vp.grad` max abs diff | `0.0003633449086919427` |
| Raw `vp.grad` max rel diff | `852.9797973632812` |

## Interpretation

The earlier receiver-output difference was not a propagating numerical
accumulation at a specific time/receiver. It was caused by the experimental
benchmark path not fully matching the production wrapper contract.

However, raw `vp.grad` still differs under observed-pressure loss even when
receiver outputs and loss match exactly. This means the remaining issue is not
receiver-output parity. It is a backward-path parity issue under the upstream
gradient produced by the observed-pressure loss.

## Decision

Do not integrate the experimental path.

Stop investigating receiver-output locations for this case: the locator found
none once the wrapper contract was matched. Continue with custom backward
diagnostics.

## Next Direction

```text
Compare production autograd and custom backward under controlled external
upstream gradients applied directly to receiver records. Use the observed-loss
upstream gradient if possible, then identify whether pressure, source
injection, free-surface, or velocity-update adjoint terms cause the raw
`vp.grad` difference.
```
