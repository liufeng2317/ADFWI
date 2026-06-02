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
