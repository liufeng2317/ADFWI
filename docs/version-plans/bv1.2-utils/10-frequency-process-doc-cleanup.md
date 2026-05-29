# 10 - Frequency Process Documentation Cleanup

## Goal

Clean `ADFWI/utils/frequency_domin_process.py` documentation now that its
current behavior is covered by contract tests.

## Scope

- `ADFWI/utils/frequency_domin_process.py`
- `docs/version-plans/bv1.2-utils/`

No FFT, filtering, plotting, or file-output behavior is changed.

## Changes

- Clarified `calculate_spectrum` input shape and return values.
- Removed unused `n_receivers` and `fs` locals from `calculate_spectrum`.
- Corrected `filter_low_frequencies_zero_phase` documentation and comments from
  low-pass wording to the actual high-pass Butterworth behavior.
- Removed unused `n_time` and `nyquist_freq` locals.
- Kept filter order, cutoff normalization, `btype="high"`, `filtfilt`, loops,
  and output allocation unchanged.

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

Stop frequency-helper cleanup unless a caller or documentation issue appears.
The next isolated utils target is legacy mute helper documentation/tests:
`offset_mute.py` and `first_arrivel_picking.py`.
