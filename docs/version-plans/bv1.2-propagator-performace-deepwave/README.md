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
| `optimization-test-guidelines.md` | Minimal test selection rules for each optimization stage |
| `implementation-change-log.md` | Compact record of design and implementation steps |
| `baselines/checkpoint10/baseline-matrix-results.md` | Current production checkpoint=10 baseline for 1/3/40 shots and 3/10 FWI iterations |
| `baselines/checkpoint-sweep-results.md` | Production checkpoint=1/5/10 speed and memory comparison |

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

Use `baselines/checkpoint10/baseline-matrix-results.md` as the main production
comparison table before and after any Deepwave-inspired implementation.

The checkpoint sweep shows that `checkpoint_segments=1` is the current speed
upper-bound reference, but it increases peak memory by about 6.82x-8.91x.
`checkpoint_segments=5` is slower than `10` and uses more memory in the current
Marmousi2 matrix, so it is not a useful default target.
