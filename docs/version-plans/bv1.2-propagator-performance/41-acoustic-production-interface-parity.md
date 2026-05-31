# Acoustic Production-Interface Parity

Date: 2026-05-31

## Purpose

This is the first post-fix interface-level gate after the `gw` view-aliasing
bug was fixed in the benchmark-only custom backward.

The test compares:

- reference: production `forward_kernel` through the normal acoustic FWI batch
  path;
- candidate: benchmark-only `experimental_forward_kernel` using the same
  validation model, survey, wavelet, damping, density, receiver geometry, and
  observed-pressure loss.

No production propagator code is changed.

## Command

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --batch-size 3 \
  --checkpoint-segments 1 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --waveform-normalize \
  --candidate-mode experimental \
  --loss-mode observed-pressure \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_forward_iteration_parity_fullshape_fixed_20260531.json
```

The temporary `output-root` directory is not committed.

## Shape

| Field | Value |
| --- | ---: |
| Shots | `3` |
| Receivers | `200` |
| `nt` | `3000` |
| `nx` | `200` |
| `nz` | `88` |
| `checkpoint_segments` | `1` |
| Device / dtype | `npu:0` / `float32` |

## Result

| Metric | Value |
| --- | ---: |
| Loss abs diff | `0.0` |
| Receiver output max abs diff | `0.0` |
| Receiver output max rel diff | `0.0` |
| Raw `vp.grad` max abs diff | `2.896413207054138e-07` |
| Raw `vp.grad` max rel diff | `0.042773060500621796` |

Timings:

| Path | Forward | Backward | Total measured |
| --- | ---: | ---: | ---: |
| Production | `5.9416 s` | `16.0118 s` | about `21.99 s` |
| Experimental prototype | `27.6953 s` | `37.4066 s` | about `65.11 s` |

The reported speedup is below `1.0` because the experimental path is still a
Python-loop benchmark prototype:

```text
forward: 0.2145x
backward: 0.4280x
total: 0.3378x
```

## Decision

The post-fix custom path now passes the interface-level output/loss gate and is
close on raw `vp.grad` in NPU float32. The remaining max relative error is
driven by near-zero reference-gradient entries; the absolute error is
`2.90e-7`.

This is a positive numerical signal, but not a production optimization yet:

- the prototype is slower than production because it loops per source in
  Python;
- it has not been implemented inside the production kernel surface;
- full FWI trajectory validation has not been run with the custom path.

## Next Direction

Stay on the acoustic custom-gradient route, but switch from formula debugging
to implementation design:

1. Design a production-facing custom backward integration point that preserves
   current `forward_kernel` inputs/outputs.
2. Avoid per-source Python loops in the production-facing design.
3. Before integration, define the acceptance gate:
   - interface parity on validation geometry;
   - raw `vp.grad` absolute tolerance around `1e-6` for NPU float32;
   - reduced FWI trajectory smoke with finite gradients.
