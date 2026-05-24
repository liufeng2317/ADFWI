# bv1.2 Low-Pass Filter Comparison

## Purpose

This note records the first validation step for replacing the legacy
`multiScaleProcessing.lpass` path with the pure torch `LowPassFilter` transform.
The goal is not to replace the FWI low-pass path immediately, but to establish a
repeatable comparison before changing inversion behavior.

## Current Implementations

- Legacy path: `ADFWI.fwi.multiScaleProcessing.lpass`
  - Uses a custom `torch.autograd.Function`.
  - Calls SciPy `butter` / `filtfilt` through NumPy arrays.
  - CPU-oriented internally because tensors are detached and converted with `.numpy()`.

- Candidate path: `ADFWI.fwi.transforms.LowPassFilter`
  - Pure torch implementation using a Hann-windowed sinc FIR kernel.
  - Uses `torch.nn.functional.conv1d` along the waveform time dimension.
  - Differentiable through torch autograd.
  - Runs on CPU and NPU in transform-level tests.

## Comparison Setup

The dedicated test file is `tests/test_lowpass_transform_comparison.py`. It uses
a synthetic waveform with a 20 Hz low-frequency component plus a 180 Hz
high-frequency component, sampled with `dt=0.001`, and compares both filters with
a 60 Hz cutoff.

Because the two filters use different boundary handling, the direct waveform
comparison ignores the edge samples and compares the trace interior.

## Observed Metrics

Initial CPU comparison on the synthetic trace showed:

- interior mean absolute difference: about `2.8e-4`;
- interior correlation: about `1.0`;
- legacy high-band amplitude ratio above 120 Hz: about `0.183`;
- torch FIR high-band amplitude ratio above 120 Hz: about `0.035`.

These values indicate that `LowPassFilter` preserves the low-frequency waveform
shape in the trace interior while attenuating the high-frequency band at least as
strongly as the legacy path for this signal.

## Tests Added

`tests/test_lowpass_transform_comparison.py` checks that:

- `LowPassFilter` matches the legacy `lpass` trace interior with correlation
  greater than `0.999`;
- interior mean absolute difference stays below `1e-3`;
- high-band attenuation is at least as strong as the legacy path;
- the torch path remains differentiable under autograd.

Existing `tests/test_data_transforms.py` also checks CPU/NPU execution for
`LowPassFilter`.

## Migration Decision

Do not replace `AcousticFWI.calculate_loss()` or `ElasticFWI.calculate_loss()`
yet. The next validation step should run acoustic and elastic mini inversion
smoke tests with a non-None `cutoff_freq` for both the legacy path and a temporary
transform-pipeline path, then compare loss, gradient norm, model update norm, and
backend behavior.
