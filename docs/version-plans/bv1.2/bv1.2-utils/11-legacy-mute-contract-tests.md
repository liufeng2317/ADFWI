# 11 - Legacy Mute Contract Tests

## Goal

Lock the direct behavior of the legacy mute helpers before documentation
cleanup.

## Scope

- `tests/test_utils_legacy_mute.py`
- `docs/version-plans/bv1.2-utils/`

No mute implementation is changed in this record.

## Covered Contracts

- `brutal_picker(trace)` uses a per-trace threshold equal to
  `0.001 * max(abs(trace))` and returns the first sample above that threshold.
- `mask(itmin, itmax, nt, length)` keeps the current sine taper branch behavior.
- `mute_arrival(trace, ...)` applies `1 - mask(...)` on the trace device.
- `apply_mute(mute_late_window, shot, dt)` preserves shot shape/dtype and uses
  the hard-coded `length = 100` loop over traces.
- `mute_offset(rcv_x, src_x, dx, waveform, distance_threshold)` zeros receivers
  with distance strictly less than `distance_threshold / dx`, keeps farther
  receivers, mutates the input waveform tensor, and returns it.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_legacy_mute.py
conda run -n adfwi python -m unittest tests/test_mute_transform_comparison.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_legacy_mute.py
git diff --check
```

## Result

- `tests/test_utils_legacy_mute.py` passed: 6 tests.
- Existing mute transform comparison tests passed: 5 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_legacy_mute.py`.
- Only existing local NPU/Ascend warnings and a legacy low-pass tensor-copy
  warning were printed.

## Documentation Follow-Up

After the contracts passed, this round also clarified the docstrings in
`offset_mute.py` and `first_arrivel_picking.py`:

- documented that `mute_offset` mutates the waveform in place and uses
  grid-index distances;
- documented the per-trace threshold in `brutal_picker`;
- documented the legacy tapered mask and late-window shot shape;
- kept formulas, function names, filename spelling, and the unused `mutetype`
  argument unchanged for compatibility.

## Next Bounded Task

Use these tests to clarify `offset_mute.py` and `first_arrivel_picking.py`
docstrings without changing legacy formulas or filenames.
