# Acoustic FWI Pressure Policy Record

This record tracks FWI-layer pressure-output policy changes. The propagator
public forward contract remains full-output by default; this file is only about
the acoustic FWI inversion loop.

## Boundary

Target:

- `ADFWI/fwi/acoustic_fwi.py`
- benchmark policy plumbing in `scripts/benchmark/acoustic_fwi_iteration_profile.py`

Do not change:

- `AcousticPropagator.forward` default output contract;
- acoustic finite-difference recurrence;
- loss function semantics;
- raw `vp.grad`.

## Trial 1: make AcousticFWI pressure-only policy automatic

Date: 2026-06-02

Hypothesis:

`AcousticFWI` constructs its misfit from pressure records only and uses pressure
illumination for gradient processing. The previously accepted
`pressure_only=True` path can therefore become the default FWI-layer behavior
without changing the public propagator forward default. Users can still force
the full-output path with `pressure_only=False`.

Implementation:

- added `resolve_acoustic_pressure_only_policy`;
- changed `AcousticFWI.forward` and `forward_closure` default from `False` to
  `"auto"`;
- `"auto"` resolves to pressure-only unless the incompatible custom-chunk expert
  path is requested;
- benchmark CLI now supports a tri-state policy:
  default auto, `--pressure-only`, and `--no-pressure-only`.

Validation:

- `py_compile ADFWI/fwi/acoustic_fwi.py scripts/benchmark/acoustic_fwi_iteration_profile.py`: passed
- backend integration parity tests for pressure-only and custom-chunk guards:
  passed
- reduced checkpoint=10 loss trajectory matched exactly:
  `6375.7919921875 -> 6006.28125 -> 5718.13330078125`
- full-record checkpoint=10 loss trajectory matched exactly:
  `74756.2578125 -> 71691.453125 -> 68891.140625`
- raw and processed gradients remained finite

Timing, steady-state average over iterations 2-3:

| Case | Metric | Full-output FWI | Auto pressure FWI | Result |
| --- | --- | ---: | ---: | ---: |
| reduced, checkpoint=10 | total iteration | 29.5813 s | 26.7011 s | +9.74% |
| reduced, checkpoint=10 | forward | 4.5536 s | 3.9228 s | +13.85% |
| reduced, checkpoint=10 | backward | 24.3816 s | 22.1281 s | +9.24% |
| full-record, checkpoint=10 | total iteration | 28.2031 s | 25.2902 s | +10.33% |
| full-record, checkpoint=10 | forward | 4.2238 s | 3.7216 s | +11.89% |
| full-record, checkpoint=10 | backward | 23.3267 s | 20.9182 s | +10.32% |

Decision:

Accepted as an FWI-layer production policy change. It promotes an already
validated pressure-only path to the standard AcousticFWI behavior while keeping
the lower-level propagator full-output API unchanged.

Next route:

Use AcousticFWI's auto policy as the new default validation baseline. Further
performance work should avoid changing AcousticPropagator defaults unless an
example or visualization workflow explicitly requires it.

## New Reduced Baseline

Date: 2026-06-02

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 10 \
  --checkpoint-segments 10
```

Configuration:

| Item | Value |
| --- | --- |
| pressure policy | `auto` |
| shots | 3 |
| batch size | 3 |
| receivers | 200 |
| nt | 3000 |
| grid | 200 x 88 |
| checkpoint segments | 10 |
| gradient processor | legacy |

Loss trajectory:

```text
6375.7919921875
6006.28125
5718.13330078125
5495.1796875
5338.6748046875
5215.0068359375
5091.37451171875
4978.30712890625
4875.275390625
4776.7587890625
```

Validation:

- all losses finite;
- all raw gradients finite;
- all processed gradients finite;
- `vp_update_norm = 8410.6171875`.

Timing:

| Metric | Average, all 10 iterations | Average, excluding first |
| --- | ---: | ---: |
| total iteration | 26.5615 s | 26.4793 s |
| forward | 3.7714 s | 3.7645 s |
| backward | 22.0717 s | 22.0680 s |
| gradient processing | 0.6404 s | 0.6404 s |

Decision:

Use this as the reduced checkpoint=10 baseline for subsequent acoustic
performance work on this branch.
