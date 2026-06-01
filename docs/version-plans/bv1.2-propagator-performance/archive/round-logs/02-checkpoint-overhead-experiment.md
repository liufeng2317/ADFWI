# Acoustic Checkpoint Overhead Experiment

Date: 2026-05-30

## Boundary

```text
Goal:
  Test whether checkpoint_segments=1 pays unnecessary checkpoint overhead in
  the acoustic differentiable forward/backward path.

Scope:
  Benchmark only. No propagator kernel behavior changed in this round.

Validation:
  Compare checkpoint vs direct step_forward calls on the same acoustic case:
  all output tensors, scalar loss, and vp gradient.

Stop:
  Stop after timing and numerical parity are recorded. Implementation is a
  separate optimization round.
```

## Method

Benchmark script:

```text
scripts/benchmark/acoustic_checkpoint_overhead.py
```

The script compares two modes inside one process:

- `checkpoint`: current behavior, using
  `ADFWI.propagator.acoustic_kernels.checkpoint(step_forward, ...)`;
- `direct`: temporary benchmark-only monkeypatch where
  `ADFWI.propagator.acoustic_kernels.checkpoint` calls `step_forward` directly.

No library code is changed by the experiment. The monkeypatch is scoped to the
benchmark process.

Compared quantities:

- `p`, `u`, `w`;
- `forward_wavefield_p`, `forward_wavefield_u`, `forward_wavefield_w`;
- scalar loss: `record["p"].pow(2).mean()`;
- `model.vp.grad`;
- forward wall time;
- backward wall time.

## NPU Command

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
  --f0 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_checkpoint_overhead_20260530.json
```

Raw JSON output is ignored by git; this document records the relevant results.

## Timing Result

Device and dtype:

```text
device: npu:0
dtype: float32
checkpoint_segments: 1
nx, nz: 100, 50
nabc: 20
nt: 800
shots: 1
receivers: 3
```

Mean over two repeated pairs:

| Mode | Forward (s) | Backward (s) | Total (s) |
| --- | ---: | ---: | ---: |
| checkpoint | 1.7408 | 7.6549 | 9.3957 |
| direct | 1.8658 | 5.1124 | 6.9782 |

Speedup:

| Metric | Direct vs checkpoint |
| --- | ---: |
| forward | noisy, not improved in mean |
| backward | about `1.50x` faster |
| total | about `1.35x` faster |

The result confirms that checkpoint overhead is material for the differentiable
acoustic path when `checkpoint_segments=1`, especially in backward.

## Numerical Result

Across both repeated pairs:

```text
loss_abs_diff: 0.0
loss_rel_diff: 0.0
p/u/w max_abs_diff: 0.0
p/u/w max_rel_diff: 0.0
forward_wavefield_p/u/w max_abs_diff: 0.0
forward_wavefield_p/u/w max_rel_diff: 0.0
vp_grad max_abs_diff: 0.0
vp_grad max_rel_diff: 0.0
```

This is a strict parity result for the measured case.

## Interpretation

The checkpoint call is unnecessary for the no-segmentation case from a
correctness perspective in this controlled experiment. It also adds a clear
backward-time cost.

The likely first code optimization is:

```text
if checkpoint_segments == 1:
    call step_forward directly
else:
    keep torch.utils.checkpoint.checkpoint(...)
```

This must be implemented carefully inside `acoustic_kernels.forward_kernel` and
validated as a separate code-change round.

## Risks Before Implementation

- Direct mode may retain more autograd graph memory than checkpoint mode.
- The benchmark case is smaller than full-record Marmousi2.
- The experiment is acoustic-only.
- Full FWI iteration may expose different memory behavior than this single-shot
  benchmark.

## Required Next Validation

Before accepting the code change:

1. run the same benchmark after implementation and confirm the direct path is
   selected only for `checkpoint_segments=1`;
2. run tiny CPU and NPU parity checks;
3. run reduced Marmousi2 forward and short inversion;
4. run full-record forward timing;
5. only then consider a 10-iteration or longer full-record inversion timing
   comparison.

