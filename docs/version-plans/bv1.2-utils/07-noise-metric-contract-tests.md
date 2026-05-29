# 07 - Noise and Metric Contract Tests

## Goal

Lock the current behavior of the small, isolated `noise.py` and
`assessment_metric.py` helpers before considering documentation or implementation
changes.

## Scope

- `tests/test_utils_noise_metrics.py`
- `docs/version-plans/bv1.2-utils/`

No noise or metric implementation is changed in this round.

## Covered Contracts

Noise helper:

- `add_gaussian_noise` preserves input shape.
- Fixed `seed` gives reproducible output.
- Current formula uses trace means over axis 1 with shape `[shot, 1, trace]`,
  `loc=trace_mean * mean_bias_factor`, scalar `scale=std_noise`, and
  `np.random.normal`.
- `seed=None` does not reset the global NumPy RNG.

Metric helpers:

- `MSE`, `RMSE`, `MAPE`, and `SNR` match current formulas for 2D inputs.
- `RMSE`, `MAPE`, and `SNR` return per-model values for batched inversion
  inputs.
- `SSIM` delegates to `skimage.metrics.structural_similarity` with the current
  per-input `data_range` rule for both 2D and batched inputs.

## Validation

```bash
conda run -n adfwi python -m unittest tests/test_utils_noise_metrics.py
conda run -n adfwi python -m unittest tests/test_utils_wavelets.py tests/test_utils_conversion.py
conda run -n adfwi python -m py_compile ADFWI/utils/*.py tests/test_utils_noise_metrics.py
git diff --check
```

## Result

- `tests/test_utils_noise_metrics.py` passed: 6 tests.
- Existing utils contract tests passed: 17 tests.
- `py_compile` passed for `ADFWI/utils/*.py` and
  `tests/test_utils_noise_metrics.py`.
- Only existing local NPU/Ascend warnings were printed.

The local `adfwi` environment does not provide `skimage`; the SSIM tests use a
small fake `structural_similarity` module to lock ADFWI's delegation contract
without adding a new dependency in this cleanup round.

## Next Bounded Task

Use these tests to decide whether `noise.py` needs a documentation-only cleanup.
Do not change metric formulas unless a benchmark/reporting requirement calls for
new definitions.
