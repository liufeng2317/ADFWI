# 21. Acoustic Segment Forward Backend

## Purpose

This step adds a narrow, opt-in backend hook for the pressure-only acoustic time-segment contract. The goal is not to change production propagation yet. It gives later fused/rematerialized/compiled segment work a stable Python dispatch point that can be parity-tested before gradients or low-level kernels are introduced.

## Code Change

- Added `backend="custom_autograd_forward"` support to `acoustic_pressure_segment` in `ADFWI/propagator/acoustic_operator.py`.
- The backend wraps the existing production `step_forward_pressure_only` segment in a `torch.autograd.Function` shell.
- Forward returns the same local segment outputs as `torch_reference`: final `p/u/w` state and `rcv_p` samples.
- Backward is intentionally unavailable and raises a clear error. This avoids silently using an incomplete gradient path.
- Added `scripts/benchmark/acoustic_segment_backend_compare.py` for segment backend timing/parity comparison.

## Validation

Command:

```bash
conda run -n adfwi pytest -q tests/test_acoustic_operator_segment.py
conda run -n adfwi python scripts/benchmark/acoustic_segment_backend_compare.py \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_segment_backend_compare_20260606.json
```

Result:

- Contract tests: `4 passed`.
- Device: `npu:0`, dtype `torch.float32`.
- Cases: 40 shots, 64 receivers, `segment_nt=4/8/16`.
- Forward parity: exact for `rcv_p`, final `p`, `u`, and `w` in all tested segment lengths.
- Finite check: passed.

| segment_nt | median speedup vs torch_reference | max abs diff |
| --- | ---: | ---: |
| 4 | 1.037x | 0.0 |
| 8 | 1.032x | 0.0 |
| 16 | 1.004x | 0.0 |

## Interpretation

The current backend is a boundary/prototype, not a real performance win. The small speedup is likely dispatch noise because it still calls the same production step function internally. The value is that the segment API can now host a future custom implementation without touching production `AcousticPropagator`.

## Next Direction

Stay on the acoustic segment main line, but do not add more shell backends. The next meaningful step must implement one real change behind this boundary:

1. fused multi-step segment forward kernel, or
2. segment-level custom backward/rematerialization policy with explicit memory and gradient parity tests.

Production integration should remain blocked until receiver output, final wavefield state, loss trajectory, and `vp` gradient parity pass on a reduced FWI case.
