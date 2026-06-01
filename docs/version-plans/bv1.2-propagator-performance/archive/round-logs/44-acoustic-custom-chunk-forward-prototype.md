# Acoustic Custom Chunk Forward Prototype

Date: 2026-05-31

## Optimization Path

Main line: Phase B, acoustic AD graph and backward cost.

Previous custom-gradient prototypes proved the one-step formulas and the
production-interface parity path, but the timestep-level custom autograd route
was not production-suitable because it still launched one
`autograd.Function.apply` per timestep. This round tests the production-facing
design from `43-acoustic-production-facing-design.md`: one custom autograd
wrapper owns a full time chunk and replays the saved recurrence in backward.

This is still a benchmark-only prototype under `scripts/benchmark`. It does not
modify `ADFWI/propagator/acoustic_kernels.py`.

## What Changed

Added:

- `scripts/benchmark/acoustic_custom_chunk_forward.py`

The script compares two paths on the same synthetic acoustic recurrence:

| Path | Meaning |
| --- | --- |
| reference | normal PyTorch autograd through the p/u/w recurrence |
| candidate | one chunk-level `torch.autograd.Function` for the full recurrence |

The candidate stores per-step p/u/w states and finite-difference intermediate
terms in forward, then runs the manually derived adjoint in reverse order. This
keeps the formula-level path already validated earlier, while removing the
per-timestep custom-autograd dispatch overhead.

## Validation Commands

Small NPU smoke:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_chunk_forward.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --steps 20 \
  --shots 1 \
  --nx 64 \
  --nz 32 \
  --nabc 16 \
  --receivers 32 \
  --loss-kind receiver-random-linear \
  --loss-components rcv_p \
  --upstream-scale 4500 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_chunk_forward_npu32_20260531.json
```

300-step NPU gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_chunk_forward.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --steps 300 \
  --shots 1 \
  --nx 64 \
  --nz 32 \
  --nabc 16 \
  --receivers 32 \
  --loss-kind receiver-random-linear \
  --loss-components rcv_p \
  --upstream-scale 4500 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_chunk_forward_300step_npu32_20260531.json
```

300-step CPU float64 formula gate:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_chunk_forward.py \
  --device cpu \
  --prefer cpu \
  --dtype float64 \
  --warmup 0 \
  --repeat 1 \
  --steps 300 \
  --shots 1 \
  --nx 64 \
  --nz 32 \
  --nabc 16 \
  --receivers 32 \
  --loss-kind receiver-random-linear \
  --loss-components rcv_p \
  --upstream-scale 4500 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_chunk_forward_300step_cpu64_20260531.json
```

Static check:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_custom_chunk_forward.py
git diff --check
```

## Results

| Case | Output diff | Loss diff | Max grad abs diff | Max grad rel diff | Forward speedup | Backward speedup | Total speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20-step NPU float32 | `0.0` | `0.0` | `1.2207e-4` | `9.2710e-6` | `5.386x` | `1.725x` | `2.852x` |
| 300-step NPU float32 | `0.0` | `0.0` | `5.8594e-3` | `7.3969e-4` | `1.740x` | `1.826x` | `1.799x` |
| 300-step CPU float64 | `0.0` | `0.0` | `1.8190e-11` | `4.5671e-13` | `1.587x` | `1.743x` | `1.689x` |

The CPU float64 gate shows the chunk-level adjoint is formula-consistent to
near machine precision. The NPU float32 gradient difference is larger because
the backward accumulation order changes, but the forward outputs and loss remain
exact. This is not yet enough to edit the production kernel; it is enough to
continue the chunk-level route instead of returning to timestep-level
prototypes.

## Decision

Continue this line, but keep it outside production until a production-interface
chunk test passes on the validation geometry.

Do not continue the previous timestep-level production route. It has a weaker
speed ceiling and repeats the Python dispatch overhead the chunk design avoids.

## Remaining Risks

- NPU float32 gradient differences are tolerance-sensitive because the custom
  backward changes accumulation order.
- The benchmark currently uses unique receiver indices. If duplicate receiver
  locations are supported, production code must use an accumulation-safe
  receiver-gradient path.
- The benchmark saves all states for the chunk. Production integration must
  decide chunk size and memory policy before replacing checkpoint behavior.

## Next Direction

Stay on Phase B and build a production-interface chunk parity harness against
the real acoustic validation wrapper, still benchmark-only. The next gate must
compare receiver records, loss, raw `vp.grad`, timing, and memory-relevant chunk
shape before any edit to `ADFWI/propagator/acoustic_kernels.py`.
