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


## Inversion Smoke Comparison

The smoke scripts now support controlled low-pass comparison without changing
default behavior:

```bash
python scripts/smoke/acoustic_mini_inversion_smoke.py --cutoff-freq 60 --lowpass-mode legacy
python scripts/smoke/acoustic_mini_inversion_smoke.py --cutoff-freq 60 --lowpass-mode transform
python scripts/smoke/elastic_mini_inversion_smoke.py --cutoff-freq 60 --lowpass-mode legacy
python scripts/smoke/elastic_mini_inversion_smoke.py --cutoff-freq 60 --lowpass-mode transform
```

`legacy` sends `cutoff_freq` into the existing FWI `calculate_loss()` low-pass
branch. `transform` uses `LowPassFilter` in the data transform pipeline and
disables the legacy cutoff branch for that run.

### Results on 2026-05-24

All runs used `cutoff_freq=60 Hz`, `dtype=float32`, one shot, one inversion
iteration, and the existing mini smoke model sizes.

| Workflow | Device | Mode | Loss | Grad norm | Update norm |
| --- | --- | --- | ---: | ---: | ---: |
| Acoustic | CPU | legacy | `3.798891725637077e-07` | `1.8225856379672223e-08` | `1.8225871324539185` |
| Acoustic | CPU | transform | `3.1958424528966134e-07` | `1.525940973579054e-08` | `1.525928020477295` |
| Acoustic | NPU | legacy | `3.7988914414199826e-07` | `1.82258599323859e-08` | `1.8225871324539185` |
| Acoustic | NPU | transform | `3.1979018899619405e-07` | `1.5259093544273128e-08` | `1.525928020477295` |
| Elastic | CPU | legacy | `0.0008560275891795754` | `4.369477755972184e-05` | `4.369435787200928` |
| Elastic | CPU | transform | `0.0006804076256230474` | `3.510879469104111e-05` | `3.5107815265655518` |
| Elastic | NPU | legacy | `0.00085602723993361` | `4.3694784835679457e-05` | `4.369436264038086` |
| Elastic | NPU | transform | `0.0006794735672883689` | `3.510589158395305e-05` | `3.510537624359131` |

### Interpretation

- CPU and NPU results are consistent within each mode, which confirms that the
  smoke scripts can compare backend behavior reproducibly.
- The transform path runs successfully on NPU and keeps gradients finite.
- The legacy path also completes on NPU for this tiny smoke case, but it still
  depends on the older SciPy/NumPy filtering implementation internally.
- The transform path is not numerically equivalent to the legacy path in the
  inversion loop: acoustic loss/gradient are roughly 16% lower, and elastic
  loss/gradient are roughly 20% lower for this cutoff. The likely causes are
  different filter design and boundary handling, especially with very short
  `nt=30` smoke traces.

## Updated Decision

Keep `multiScaleProcessing.lpass` as the default FWI low-pass path for now.
`LowPassFilter` remains a validated, differentiable, CPU/NPU-capable candidate,
but it should not replace the legacy path until we either:

1. tune the FIR design and boundary handling to better match legacy behavior; or
2. intentionally accept the new filter as a changed numerical method and document
   the expected inversion differences.

The next useful validation is to repeat the comparison on longer traces and with
multiple cutoff frequencies, because the current mini smoke length magnifies
boundary effects.
