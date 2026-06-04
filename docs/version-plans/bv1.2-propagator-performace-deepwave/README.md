# Deepwave-Inspired Propagator Performance Plan

This folder records the next performance direction after the bv1.2
propagator cleanup. It is based on reading the local Deepwave source under
`external/deepwave`, but it does not copy Deepwave code.

The goal is to identify which design ideas can improve ADFWI's acoustic
propagator without breaking numerical precision, autograd gradients, or the
existing FWI interface.

## Active Documents

| Document | Purpose |
| --- | --- |
| `00-deepwave-inspired-outline.md` | Top-to-bottom optimization outline and implementation phases |
| `01-acoustic-operator-contract.md` | Detailed ADFWI acoustic operator contract and first implementation boundary |
| `deepwave-method-map.md` | What Deepwave does and what ADFWI can learn from it |
| `performance-test-matrix.md` | Required tests before any Deepwave-inspired change is promoted |
| `implementation-change-log.md` | Compact record of design and implementation steps |
| `baselines/baseline-matrix-results.md` | Current production baseline for 1/3/40 shots and 3/10 FWI iterations |

## Boundary

- This is a design and evaluation plan, not an implementation branch.
- `external/deepwave` is a local reference repository and is ignored by git.
- Production changes should start only after the test matrix and success gates
  are accepted.
- The first implementation target should be acoustic only. Elastic should wait
  until acoustic proves the design.

## Current Baseline

The current production baseline was measured on the full-record Marmousi2
validation geometry with `checkpoint_segments=10`, `dtype=float32`, and
`device=npu:0`.

Use `baselines/baseline-matrix-results.md` as the main comparison table before
and after any Deepwave-inspired implementation.
