# Acoustic Experimental Forward Iteration Parity

Date: 2026-05-31

## Purpose

This test moves the benchmark-only `experimental_forward_kernel` from a tiny
kernel parity check into a reduced Marmousi2 FWI-style iteration.

It still does not change production `ADFWI/propagator` code. The test compares
one production iteration against one experimental-forward iteration with:

- the same reduced Marmousi2 model and survey construction;
- 3 shots, full-batch pressure loss;
- `save_forward_wavefield=False`;
- `grad_forw_illumination=False`;
- waveform normalization enabled;
- raw `vp.grad` comparison before gradient processing.

## Change

Added:

- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`

The script builds the reduced Marmousi2 validation state and compares:

- receiver `p/u/w` records;
- scalar pressure loss;
- raw `vp.grad`;
- forward, backward, and total timing.

The experimental path loops over sources internally because the current
`experimental_forward_kernel` supports one source at a time. The records are
then concatenated and evaluated as one full-batch loss.

## Command

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
  --waveform-normalize \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_standard_fullsize \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_forward_iteration_parity_fullsize_20260531.json
```

## Result

Reference record:

- `acoustic_experimental_forward_iteration_parity_fullsize_20260531.json`

| Metric | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Model | `nx=200`, `nz=88`, `nabc=30` |
| Time steps | `3000` |
| Shots / receivers | `3 / 200` |
| Loss absolute difference | `0.0` |
| Output maximum absolute difference | `6.51925802230835e-09` |
| Output maximum relative difference | `91.71505737304688` |
| Raw `vp.grad` reference norm | `0.877909779548645` |
| Raw `vp.grad` candidate norm | `0.8779159784317017` |
| Raw `vp.grad` maximum absolute difference | `0.00036325614200904965` |
| Raw `vp.grad` maximum relative difference | `853.6019897460938` |
| Forward speedup | `0.20216920757965348x` |
| Backward speedup | `0.43160782528016484x` |
| Total speedup | `0.3357941296789839x` |

## Interpretation

This gate is not accepted as a production-integration pass.

The receiver records are very close in absolute error and the loss is exactly
the same at the reported precision, but the raw `vp.grad` maximum absolute
difference is too large for a core differentiable propagator change. The
experimental path is also slower at this level because it loops over sources in
Python and is still a benchmark scaffold, not a fused production path.

The high output relative difference is caused by near-zero receiver samples,
but the gradient absolute difference is sufficient by itself to reject this
gate.

## Decision

Do not integrate `experimental_forward_kernel` into production propagator code.

Continue only with gradient parity diagnosis. The next task should identify
which part of the full-size FWI path introduces the raw `vp.grad` difference:

- source batching versus per-source loop;
- waveform normalization sensitivity;
- coefficient construction or padding difference;
- accumulated floating-point ordering over 3000 time steps;
- source/free-surface adjoint contribution in the custom backward.

## Next Direction

```text
Stop expanding the experimental path. Add a focused gradient-difference
localization probe that compares production and experimental gradients for
progressively larger settings: no normalization, one shot, three shots, short
nt, full nt. Only if raw `vp.grad` returns to tiny-case tolerance should
production-facing work resume.
```
