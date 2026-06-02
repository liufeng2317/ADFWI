# No-Checkpoint Direct Return Gate

Date: 2026-06-02

## Goal

Test one bounded production acoustic hot-path idea:

```text
checkpoint_segments == 1
```

The hypothesis was that `forward_kernel` could directly return the full outputs
from `step_forward` and avoid the outer receiver/wavefield allocation plus copy.

## Result

The code change was tested and then reverted.

Validation command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 2 \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5
```

Numerical result:

```text
output max_abs_diff: 0.0
loss_abs_diff: 0.0
vp_grad max_abs_diff: 0.0
```

Timing summary:

| Path | Before mean | After mean | Decision |
| --- | ---: | ---: | --- |
| forward seconds | `1.6232` | `1.6625` | slower |
| total seconds | `6.2665` | `6.4583` | slower |

## Decision

Do not keep this production code change.

The optimization is numerically safe, but it does not improve the measured NPU
case. The production kernel was restored to the previous implementation.

Raw reports:

- `archive/results/acoustic_no_checkpoint_direct_return_before_20260602.json`
- `archive/results/acoustic_no_checkpoint_direct_return_after_20260602.json`

## Next Direction

Continue from the summary boundary: only production acoustic PyTorch hot-path
changes with a clear performance mechanism and full forward/loss/gradient
comparison.

