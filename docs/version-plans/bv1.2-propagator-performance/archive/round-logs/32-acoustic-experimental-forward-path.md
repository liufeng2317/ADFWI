# Acoustic Experimental Forward Path

Date: 2026-05-31

## Purpose

This step creates an explicit opt-in experimental acoustic forward path for
benchmark scripts only. It does not change production `ADFWI/propagator` code
and is not imported by user examples.

The goal is to stop embedding the custom-gradient recurrence inside one parity
script and instead provide a clear experimental entry point for the next
validation gates.

## Change

Added:

- `scripts/benchmark/acoustic_experimental_forward.py`

Updated:

- `scripts/benchmark/acoustic_custom_kernel_parity_probe.py`

The new function is:

```python
experimental_forward_kernel(...)
```

It mirrors the small subset of the production acoustic `forward_kernel` needed
for benchmark parity:

- one source;
- receiver `p/u/w` records;
- source injection;
- free-surface boundary write;
- padded `v/rho` and `alpha/kappa` coefficient construction;
- raw gradient flow back to `v`.

It explicitly rejects unsupported cases, including multiple source locations
and forward-wavefield summaries.

## Command

```bash
timeout 480s conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --nx 32 \
  --nz 24 \
  --nabc 8 \
  --nt 20 \
  --receivers 16 \
  --source-depth 1 \
  --receiver-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_experimental_forward_parity_20260531.json
```

## Result

Reference record:

- `acoustic_experimental_forward_parity_20260531.json`

| Metric | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Model | `nx=32`, `nz=24`, `nabc=8` |
| Time steps | `20` |
| Receivers | `16` |
| Loss absolute difference | `0.0` |
| Output maximum absolute difference | `0.0` |
| Output maximum relative difference | `0.0` |
| `v.grad` maximum absolute difference | `4.1359030627651384e-25` |
| `v.grad` maximum relative difference | `4.1359030458244794e-13` |
| Forward mean speedup | `1.284789789163255x` |
| Backward mean speedup | `2.0879583813199356x` |
| Total mean speedup | `1.7687082231382452x` |

CPU smoke also passed with exact receiver/loss parity and `v.grad` numerical
noise only.

## Interpretation

The experimental forward entry point preserves the tiny-case production
interface parity already observed in the previous script-local prototype. This
is an organization and validation milestone, not a production optimization.

The result keeps the custom-gradient route viable and makes the next gate
practical: call the experimental path from a reduced FWI-style benchmark
without touching normal `AcousticPropagator` behavior.

## Decision

Continue to the reduced Marmousi2 parity gate. Do not integrate the
experimental path into production examples or default propagator APIs yet.

## Next Direction

```text
Build a reduced Marmousi2 iteration parity benchmark that swaps only the
forward call to `experimental_forward_kernel`, then compares receiver records,
loss, raw `vp.grad`, and iteration timing against the production path.
```
