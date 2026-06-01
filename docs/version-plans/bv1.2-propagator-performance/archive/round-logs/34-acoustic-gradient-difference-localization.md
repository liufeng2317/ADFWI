# Acoustic Gradient Difference Localization

Date: 2026-05-31

## Purpose

The previous reduced Marmousi2 FWI-style parity gate failed because raw
`vp.grad` differed by `3.63e-4`. This record localizes that difference before
any further integration work.

The rule for this phase is strict: do not expand the experimental path while
raw gradient parity is not explained.

## Checks

### 1. Waveform Normalization

Command:

```bash
timeout 3600s conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --checkpoint-segments 1 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --no-waveform-normalize \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullsize_nonorm \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_forward_iteration_parity_fullsize_nonorm_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Loss diff | `0.0` |
| Output max abs diff | `6.51925802230835e-09` |
| Raw `vp.grad` max abs diff | `0.00036325445398688316` |
| Raw `vp.grad` max rel diff | `838.3789672851562` |

Conclusion: waveform normalization is not the cause. Removing normalization
does not materially reduce the raw gradient difference.

### 2. Production Repeat Determinism

Command:

```bash
timeout 3600s conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --checkpoint-segments 1 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --no-waveform-normalize \
  --candidate-mode production \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/production_iteration_parity_fullsize_nonorm \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_production_iteration_repeat_fullsize_nonorm_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Loss diff | `0.0` |
| Output max abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `0.0` |
| Raw `vp.grad` max rel diff | `0.0` |

Conclusion: the failed parity is not caused by NPU nondeterminism or by the
benchmark script rebuilding the production state twice.

### 3. Direct Kernel, Long Single-Source Recurrence

Command:

```bash
timeout 3600s conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --nx 200 \
  --nz 88 \
  --nabc 30 \
  --nt 3000 \
  --receivers 200 \
  --source-depth 1 \
  --receiver-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_fullsize_nt3000_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Loss diff | `0.0` |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `1.1309109937248425e-26` |

Conclusion: long `nt=3000` recurrence by itself is not the cause.

### 4. Direct Kernel, Multi-Source Batch Versus Per-Source Loop

Command:

```bash
timeout 3600s conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nabc 30 \
  --nt 3000 \
  --receivers 200 \
  --source-depth 1 \
  --receiver-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_fullsize_shot3_nt3000_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Loss diff | `0.0` |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `4.1359030627651384e-25` |

Conclusion: production batched sources versus experimental per-source loop is
not the cause when the source wavelet and loss are the simple direct-kernel
benchmark setup.

### 5. Validation Source/Survey With Synthetic Energy Loss

Command:

```bash
timeout 3600s conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --checkpoint-segments 1 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --no-waveform-normalize \
  --loss-mode synthetic-energy \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullsize_synthetic_energy \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_forward_iteration_parity_synthetic_energy_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Loss diff | `0.0` |
| Output max abs diff | `6.51925802230835e-09` |
| Raw `vp.grad` max abs diff | `3.36952687973735e-14` |
| Raw `vp.grad` max rel diff | `0.005578695330768824` |

Conclusion: with the validation source/survey/model but direct synthetic-energy
loss, raw gradient parity is acceptable. The large `3.63e-4` difference appears
only when using observed-pressure residual loss.

## Decision

The experimental path remains blocked from production integration.

The most likely current explanation is:

```text
The experimental path has a tiny receiver-output absolute difference
(`6.52e-09`) under the validation source/survey setup. Direct synthetic-energy
loss does not amplify it, but observed-pressure residual loss does, producing a
raw `vp.grad` difference around `3.63e-4`.
```

This means the next task is not performance integration. The next task is to
remove or explain the `6.52e-09` receiver-output difference under the validation
source/survey setup.

## Next Direction

```text
Localize the receiver-output difference under the validation source/wavelet:
compare production and experimental receiver records by shot, component, time
index, receiver index, and source wavelet value. Continue only if output parity
returns to tiny-kernel tolerance under the validation source/survey setup.
```
