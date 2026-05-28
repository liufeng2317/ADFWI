# 115. Torch Gradient Status And Next Direction

## Goal

Record the current decision after the Marmousi2 NPU gradient-processor
validation sequence, then stop expanding single-path gradient tests. The goal is
to make the next optimization direction explicit: keep `TorchGradProcessor`
opt-in and move subsequent work back to broader framework/performance
optimization.

## Decision

`TorchGradProcessor` is now considered **NPU validated as an opt-in path** for
the current Marmousi2 gates, but it should **not replace the default legacy
`GradProcessor` yet**.

Reasons:

1. The legacy path is the historical numerical reference.
2. The torch path has passed focused parity, stage diagnostics, one-step
   Marmousi2 gates, and 3-shot/10-iteration Marmousi2 gates.
3. NPU float32 smoothing still has a known device-level `conv2d` drift relative
   to CPU/SciPy, even though real-case trajectory drift is negligible in the
   tested gates.
4. Default-path migration is a user-facing numerical-policy decision, not only a
   code cleanup.

## Validation Evidence

The current validation chain is:

| Record | Gate | Result |
| --- | --- | --- |
| 107 | Focused legacy-vs-torch gradient processor parity | Passed with explicit tolerances. |
| 108 | Cheap timing/parity benchmark | CPU strict passed; NPU strict exposed smoothing/illumination drift. |
| 109 | Stage diagnostics | Localized NPU drift to float32 `conv2d` smoothing. |
| 110 | NPU tolerance profile | Added explicit `npu-float32` benchmark profile; strict remains default. |
| 111 | Marmousi2 real-case opt-in wiring | Mask-only one-step NPU gate matched legacy. |
| 112 | Marmousi2 one-step stress gates | Smoothing and illumination stress gates matched at loss/update scale. |
| 113 | 3-shot/10-iteration smoothing trajectory | Stable; final-loss relative drift about `3e-6`. |
| 114 | 3-shot/10-iteration illumination trajectory | Stable; final loss identical and update-norm relative drift about `1e-7`. |

## Current User Guidance

- Keep the default gradient processor on `legacy`.
- Use `--gradient-processor torch` only when intentionally validating or
  benchmarking the torch-native path.
- For NPU micro-benchmarks, use `--tolerance-profile npu-float32` only when the
  goal is to accept known NPU float32 smoothing drift. The benchmark default is
  still strict.
- Do not keep adding one-step gradient-processor tests unless a new code change
  affects gradient post-processing.

## Optimization Path Change

The gradient-processor branch has reached a convergence point. Subsequent
optimization should prioritize one of these:

1. **Operator-level performance profiling** on the fixed 3-shot Marmousi2 NPU
   baseline, because Python profiling already showed most time is in autograd
   backward and checkpointed propagator execution.
2. **Data-contract/preparation cleanup** in `ADFWI.fwi.data`, because the public
   facade is now stable and internals already use owner modules.
3. **Example/preset configuration consolidation**, so repeated real-case gates
   are described by reusable configs rather than expanding CLI options further.

## Validation Result

This step is documentation/status only. Validation commands:

```bash
rg -n "NPU validated as an opt-in path|TorchGradProcessor.*opt-in|operator-level profiling|Data-contract/preparation cleanup" docs/version-plans/bv1.2 docs/backend-usage.md scripts/examples/README.md tests/full_cases/README.md

git diff --check
```

## Next Direction

Start the next optimization from `ADFWI.fwi.data` or profiling, not from more
gradient-processor trajectory tests. If the next turn focuses on the currently
open files, a reasonable target is auditing `ADFWI/fwi/data/preparation.py` and
its neighbors for duplicated validation/shape logic that can be tightened
without changing numerical behavior.
