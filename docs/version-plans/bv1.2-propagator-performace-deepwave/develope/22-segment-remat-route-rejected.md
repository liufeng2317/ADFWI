# 22. Segment Remat Route Rejected

## Purpose

The previous step added an opt-in pressure-segment backend boundary. This step tested whether a Python `torch.autograd.Function` segment rematerialization backend could become the next memory-saving implementation path.

## Attempted Direction

The candidate backend would run `step_forward_pressure_only` under `torch.no_grad()` in forward, then rerun the same short segment in backward to compute gradients. This is the same high-level idea as checkpoint/rematerialization: save less internal graph state and pay recomputation during backward.

## Result

The route is rejected for now.

Observed behavior:

- Forward parity against `torch_reference` is exact.
- Direct `torch_reference` backward for a standalone segment fails under several gradient cases because the TorchScript recurrence updates `p/u/w` in-place.
- A remat backward can compute some isolated gradients, but it cannot robustly support the full state-plus-coefficient gradient set needed for chaining pressure segments.
- In particular, when `p/u/w` state gradients are needed across segment boundaries, the recomputed TorchScript graph hits PyTorch in-place version errors.

This means the backend would not be safe as a production FWI gradient path. Keeping it as a public segment backend would be misleading.

## Validation Performed

Command:

```bash
conda run -n adfwi pytest -q tests/test_acoustic_operator_segment.py
```

After removing the invalid remat backend from the active segment backend set, the existing segment tests pass:

- `4 passed`
- NPU environment warnings only; no test failure.

## Decision

Do not continue Python segment-remat around the existing TorchScript `step_forward_pressure_only` recurrence.

The useful conclusion is narrower and stronger: if we want checkpoint-like memory behavior with better speed, the segment implementation itself must avoid the current in-place autograd conflict. That points to either:

1. a functional pressure segment recurrence written specifically for remat/autograd, or
2. a lower-level fused/custom operator with a matching custom backward.

## Next Direction

Move to a functional mini-segment prototype, not production code:

- build a tiny non-in-place pressure segment for `segment_nt=2/4` on CPU/NPU;
- verify receiver and state parity against `step_forward_pressure_only`;
- verify gradients for `p/u/w/src_v/alpha` against autograd;
- only then benchmark whether the functional form is too slow or can be compiled/fused.

If the functional prototype is much slower, stop the Python route and focus on low-level fused operator design.
