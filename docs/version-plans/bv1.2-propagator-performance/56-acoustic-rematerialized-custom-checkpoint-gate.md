# 56. Acoustic Rematerialized Custom Checkpoint Gate

Date: 2026-05-31

## Purpose

Test whether the new rematerialized acoustic custom chunk prototype can target
the real `checkpoint_segments > 1` use case.

This is different from the previous high-memory custom chunk path:

- previous custom chunk: saves chunk-internal states in forward;
- rematerialized custom chunk: saves chunk boundary states and recomputes
  chunk internals during backward.

The prototype remains experimental and is not wired into the default
`AcousticPropagator.forward` path.

## Implementation

Added:

- `ADFWI/propagator/acoustic_custom_kernels.py`
  - `_RematerializedCustomChunkForward`
  - `rematerialized_custom_chunk_forward_kernel`
- `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`
  - `experimental-remat-chunk` candidate mode
  - peak allocated memory reporting for reference/candidate runs

The rematerialized custom backward reuses the already validated one-step manual
adjoint, but reconstructs chunk-local states inside backward rather than saving
them during forward.

## Validation

Compile gate:

```bash
conda run -n adfwi python -m py_compile \
  ADFWI/propagator/acoustic_custom_kernels.py \
  scripts/benchmark/acoustic_experimental_forward_iteration_parity.py
```

### Small NPU Synthetic-Energy Gate

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 64 \
  --nz 32 \
  --nt 300 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --candidate-mode experimental-remat-chunk \
  --loss-mode synthetic-energy \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/remat_chunk_synthetic_nt300 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_synthetic_energy_gate_20260531.json
```

Result:

| Metric | Value |
| --- | ---: |
| Receiver output max abs diff | 0.0 |
| Loss abs diff | 0.0 |
| Raw `vp.grad` max abs diff | 1.7763568394002505e-15 |
| Backward speedup | 1.450x |
| Total speedup | 1.238x |

### Small NPU Observed-Pressure Gate

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 64 \
  --nz 32 \
  --nt 300 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode experimental-remat-chunk \
  --loss-mode observed-pressure \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/remat_chunk_synthetic_nt300 \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_observed_pressure_gate_20260531.json
```

Result:

| Metric | Value |
| --- | ---: |
| Receiver output max abs diff | 0.0 |
| Loss abs diff | 0.0 |
| Raw `vp.grad` max abs diff | 6.984919309616089e-08 |
| Backward speedup | 1.523x |
| Total speedup | 1.441x |

### Fullshape Reduced Observed-Pressure Memory Gate

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 10 \
  --shots 3 \
  --batch-size 3 \
  --nx 200 \
  --nz 88 \
  --nt 3000 \
  --no-save-forward-wavefield \
  --no-grad-forw-illumination \
  --waveform-normalize \
  --candidate-mode experimental-remat-chunk \
  --loss-mode observed-pressure \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/experimental_forward_iteration_parity_fullshape \
  --result-json docs/version-plans/bv1.2-propagator-performance/acoustic_remat_chunk_fullshape_observed_pressure_memory_gate_20260531.json
```

Result:

| Metric | Value |
| --- | ---: |
| Receiver output max abs diff | 0.0 |
| Loss abs diff | 0.0 |
| Raw `vp.grad` max abs diff | 2.738088369369507e-07 |
| Raw `vp.grad` max rel diff | 0.10086797922849655 |
| Production checkpoint peak allocated | 292.98 MiB |
| Remat candidate peak allocated | 810.92 MiB |
| Remat / production peak allocated | 2.768x |
| Forward ratio | 0.857x |
| Backward speedup | 1.344x |
| Total speedup | 1.229x |

## Decision

The rematerialized prototype passes the first numerical gates and gives a real
speedup for `checkpoint_segments=10`. It also greatly improves the memory
positioning compared with the previous high-memory custom chunk path:

```text
high-memory custom chunk: 28.92x production checkpoint peak memory
rematerialized custom chunk: 2.77x production checkpoint peak memory
```

However, it is still not memory-equivalent to production checkpoint. It cannot
be promoted into the default path yet.

## Next Direction

Continue this line only if the next step targets memory, not another speed-only
benchmark.

Most likely next task:

```text
Reduce rematerialized backward peak memory by avoiding full chunk-local state
lists during backward, or prove that a Python-level rematerialized implementation
cannot match torch checkpoint memory without a lower-level fused implementation.
```

Do not wire `experimental-remat-chunk` into public `AcousticFWI.forward` until
the memory ratio is much closer to production checkpoint and full-record short
FWI has passed.
