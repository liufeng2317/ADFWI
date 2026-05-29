# 09 - Frequency Process Contract Tests

## Goal

Lock the current behavior of `ADFWI/utils/frequency_domin_process.py` before
documentation or implementation cleanup.

## Scope

- `tests/test_utils_frequency_process.py`
- `docs/version-plans/bv1.2-utils/`

No frequency-processing implementation is changed in this round.

## Covered Contracts

- `calculate_spectrum(rcv, dt)`:
  - expects receiver-time data with shape `[receiver, time]`;
  - computes `np.fft.fft(..., axis=1)`;
  - returns the first `n_samples // 2` frequency bins;
  - returns amplitude as `abs(fft)` and power as `amplitude**2`.

- `filter_low_frequencies_zero_phase(data, dt, cutoff_freq)`:
  - expects data with shape `[shot, time, receiver]`;
  - preserves shape and dtype;
  - returns finite values;
  - acts as a high-pass filter in the current implementation, strongly
    attenuating a 2 Hz component while preserving a 40 Hz component for the
    tested signal.

- Plot helpers:
  - `plot_spectrum`, `plot_frequency_distribution`, and `plot_filtered_data`
    run with `show=False`;
  - each can save a PNG output on small synthetic arrays.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_frequency_process.py
conda run -n adfwi python -m unittest tests/test_utils_noise_metrics.py tests/test_utils_wavelets.py tests/test_utils_conversion.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_frequency_process.py
git diff --check
```

## Result

- `tests/test_utils_frequency_process.py` passed: 3 tests.
- Existing utils contract tests passed: 23 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_frequency_process.py`.
- Only existing local NPU/Ascend warnings and Matplotlib/PyParsing deprecation
  warnings were printed.

## Next Bounded Task

Use these tests to support a small documentation cleanup in
`frequency_domin_process.py`, especially correcting the low-pass/high-pass
wording and removing unused locals.
