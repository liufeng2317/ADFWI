# 08 - Noise Docs and Frequency Contract Audit

## Goal

Clean the documentation of the already-tested `noise.py` helper, then audit
`frequency_domin_process.py` contracts before adding tests or changing code.

## Scope

- `ADFWI/utils/noise.py`
- `ADFWI/utils/frequency_domin_process.py` audit only
- `docs/version-plans/bv1.2-utils/`

No frequency-processing implementation is changed in this round.

## Noise Documentation Changes

- Replaced the stale author/file header with a concise module docstring.
- Removed the old commented-out noise implementation.
- Clarified `add_gaussian_noise` input shape, random seed behavior, trace-mean
  bias shape, and return shape.
- Kept the current formula and global NumPy RNG behavior unchanged.

## Frequency Contract Audit

Current functions:

| Function | Current contract | Notes |
| --- | --- | --- |
| `calculate_spectrum(rcv, dt)` | Expects `rcv` with shape `[receiver, time]`; computes FFT along axis 1; returns positive frequencies, amplitude spectrum, and power spectrum. | Returns `n_samples // 2` frequency bins and omits negative frequencies. |
| `filter_low_frequencies_zero_phase(data, dt, cutoff_freq=5)` | Expects `data` with shape `[shot, time, receiver]`; applies a 6th-order Butterworth high-pass filter with `signal.filtfilt` to each trace; returns `np.zeros_like(data)` filled with filtered traces. | Name says low frequencies are removed; implementation uses `btype="high"`. |
| `plot_spectrum(...)` | Plots first receiver amplitude and power spectra. | Plot helper only; no numerical pipeline ownership. |
| `plot_frequency_distribution(...)` | Sums amplitude spectrum over receivers and plots total frequency distribution. | Plot helper only. |
| `plot_filtered_data(...)` | Plots one shot before/after filtering. | Plot helper only. |

Observed cleanup candidates:

- `calculate_spectrum` has unused locals (`n_receivers`, `fs`).
- `filter_low_frequencies_zero_phase` has an unused `nyquist_freq` local.
- Plot helpers use mutable default lists for limits, but do not mutate them.
- The filename `frequency_domin_process.py` is historically named and should not
  be renamed without import-surface migration.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_noise_metrics.py
conda run -n adfwi python -m unittest tests/test_utils_wavelets.py tests/test_utils_conversion.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_noise_metrics.py
git diff --check
```

## Result

- `tests/test_utils_noise_metrics.py` passed: 6 tests.
- Existing wavelet/conversion contract tests passed: 17 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_noise_metrics.py`.
- Only existing local NPU/Ascend warnings were printed.

## Next Bounded Task

Add `frequency_domin_process.py` contract tests for:

- FFT frequency bin shape and amplitude/power values on a small known signal;
- high-pass filtering shape/dtype/finite output and expected attenuation of a
  low-frequency sinusoid;
- plot helpers running with `show=False` on small arrays.
