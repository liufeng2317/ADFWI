# Pressure Segment Integration Test

## Focus

This round tested whether the single-step custom pressure-update backward can
still help when embedded into a multi-time-step pressure-only segment.

## What Was Implemented

Added an opt-in non-scripted pressure-only segment prototype:

```text
step_forward_pressure_only_custom_update
  |
  +-- custom pressure update
  +-- production-equivalent source injection
  +-- production-equivalent velocity updates
  +-- production-equivalent receiver pressure sampling
```

The production TorchScript `step_forward_pressure_only` and
`AcousticPropagator.forward` were not changed.

## Why A Snapshot Was Needed

The first segment run failed because later time steps mutate `p/u/w` in-place.
The custom pressure-update backward had saved references to those tensors, so
PyTorch correctly reported an in-place version conflict.

The fix was to save `p/u/w` snapshots inside the custom update. This restores
correctness but increases memory traffic and removes most of the single-step
benefit.

## Tests

Commands:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_pressure_segment_compare.py \
  --device npu:0 --nx 64 --nz 32 --nt 120 --shots 3 --receivers 64 --nabc 4 \
  --repeat 3 --warmup 1 \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_pressure_segment_compare_20260605.json

conda run -n adfwi python scripts/benchmark/acoustic_pressure_segment_compare.py \
  --device npu:0 --nx 64 --nz 32 --nt 120 --shots 40 --receivers 64 --nabc 4 \
  --repeat 3 --warmup 1 \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_pressure_segment_compare_40shot_20260605.json
```

## Results

| Case | Forward speed | Backward speed | Total speed | Parity |
| --- | ---: | ---: | ---: | --- |
| `64x32 nt=120`, 3 shots | `0.85x` | `1.14x` | `1.05x` | `rcv_p` exact; `vp.grad` max abs `2.00e-15` |
| `64x32 nt=120`, 40 shots | `0.85x` | `1.03x` | `0.97x` | `rcv_p` exact; `vp.grad` max abs `4.00e-15` |

## Decision

Do not promote the Python custom pressure-update segment into the FWI loop.

Reason:

- single-step custom backward is promising;
- multi-step Python integration needs snapshots because of in-place recurrence;
- snapshot memory traffic removes the performance benefit;
- the 40-shot case is slightly slower overall.

## Next Step

Move the pressure update idea to a real lower-level implementation:

```text
compiled pressure update / compiled time segment
  |
  +-- owns p/u/w state inside the kernel
  +-- avoids PyTorch in-place version conflicts
  +-- reduces graph nodes without saving Python-level snapshots
```

This is closer to Deepwave's actual design than Python-level custom autograd.
