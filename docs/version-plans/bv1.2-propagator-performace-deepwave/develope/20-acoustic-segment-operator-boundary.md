# Acoustic Segment Operator Boundary

## Focus

This round moved the short-segment contract from a benchmark-only idea into an
explicit opt-in operator boundary.

Production propagator code is still unchanged. `AcousticPropagator` does not
use this path by default.

## What Changed

Added to `ADFWI/propagator/acoustic_operator.py`:

- `AcousticPressureSegmentConfig`
- `AcousticPressureSegmentInputs`
- `AcousticPressureSegmentOutput`
- `acoustic_pressure_segment(..., backend="torch_reference")`

The reference backend delegates to production `step_forward_pressure_only` and
returns:

```text
p, u, w, rcv_p
```

This fixes the boundary for future fused segment implementations.

## Validation

Added:

```text
tests/test_acoustic_operator_segment.py
```

Test command:

```bash
conda run -n adfwi pytest -q tests/test_acoustic_operator_segment.py
```

Result:

```text
2 passed
```

The test compares a full pressure-only run with two `segment_nt=4` calls over
`nt=8` on a small CPU case:

| Quantity | Result |
| --- | --- |
| `rcv_p` | exact |
| final `p` | exact |
| final `u` | exact |
| final `w` | exact |
| compiled backend | explicitly unavailable |

## Boundary

This is not a performance optimization yet. It is the interface that prevents
future compiled/custom segment work from changing semantics by accident.

Current active backend:

```text
torch_reference
```

Reserved backend:

```text
compiled
```

## Next Step

Implement a benchmark-only experimental segment backend behind this boundary,
then compare it against `torch_reference` for:

1. receiver output parity;
2. final `p/u/w` parity;
3. forward timing for `segment_nt=4/8/16`.

Backward/gradient policy remains out of scope until forward parity and timing
are meaningful.
