# Acoustic Targeted Backward Exclusions

Date: 2026-05-31

## Purpose

This record replaces broad scanning with high-probability exclusion tests for
the remaining raw `vp.grad` mismatch.

Current known failure:

- receiver outputs match exactly;
- observed-pressure loss upstream gradient matches exactly;
- applying the same upstream gradient still gives raw `vp.grad` max abs diff
  around `3.63e-4`.

## Targeted Tests

### 1. Free-Surface Off

Command:

```bash
timeout 4800s conda run -n adfwi python scripts/benchmark/acoustic_observed_loss_upstream_probe.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --nabc 30 \
  --checkpoint-segments 1 \
  --no-free-surface \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/observed_loss_upstream_probe_no_freesurface \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_observed_loss_upstream_no_freesurface_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Receiver output max abs diff | `0.0` |
| Receiver upstream max abs diff | `0.0` |
| Raw `vp.grad` max abs diff | `0.00038442015647888184` |
| Raw `vp.grad` max rel diff | `13.645998001098633` |

Conclusion: free-surface adjoint is not the primary cause. Disabling
free-surface does not remove the mismatch.

### 2. Direct Pressure-Only Energy Loss

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
  --loss-components p \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_fullsize_shot3_p_only_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `4.1359030627651384e-25` |

Conclusion: pressure-only receiver loss is not enough to trigger the mismatch.

### 3. Direct Random Pressure Upstream, `nt=300`

Command:

```bash
timeout 1800s conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 3 \
  --nx 200 \
  --nz 88 \
  --nabc 30 \
  --nt 300 \
  --receivers 200 \
  --source-depth 1 \
  --receiver-depth 1 \
  --source-scale 1e-4 \
  --loss-components p \
  --loss-kind random-linear \
  --upstream-scale 1e-6 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_random_p_nt300_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `3.5998900258307764e-21` |

Conclusion: arbitrary pressure upstream at moderate time length is not enough
to trigger the mismatch.

### 4. Direct Random Pressure Upstream, `nt=3000`

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
  --loss-components p \
  --loss-kind random-linear \
  --upstream-scale 1e-6 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_kernel_parity_random_p_nt3000_20260531.json
```

Result:

| Metric | Value |
| --- | --- |
| Output max abs diff | `0.0` |
| `v.grad` max abs diff | `4.0234064994579266e-19` |

Conclusion: long-time arbitrary pressure upstream is also not enough to trigger
the mismatch.

## Decision

The following high-probability causes are now excluded:

- free-surface adjoint as primary cause;
- pressure-only receiver output as a class;
- arbitrary random pressure upstream;
- long-time random pressure upstream.

The remaining likely cause is a property specific to the observed-pressure
upstream gradient itself, such as its amplitude scale, sparsity, or
normalization-induced distribution.

## Next Direction

```text
Measure the observed receiver-upstream distribution and replay that exact
upstream through the direct kernel parity harness, outside the FWI wrapper.
If the direct replay fails, the issue is custom backward under that upstream
distribution. If it passes, the issue is in the FWI model/rho refresh or
parameter dependency path.
```
