# Checkpoint Backward Recompute Record

This record tracks the current acoustic checkpoint/backward optimization branch.
Only accepted results should be summarized in `performance-change-log.md`; this
file keeps the detailed trial evidence.

## Boundary

Target:

- `ADFWI/propagator/acoustic_kernels.py`
- acoustic production path
- `checkpoint_segments > 1`
- PyTorch autograd/checkpoint replay cost

Do not change:

- finite-difference update equations;
- receiver output shape/order;
- loss inputs;
- raw `vp.grad`;
- default output contract.

Acceptance gate:

- waveform/loss/raw-gradient parity;
- real FWI iteration timing, not isolated kernel timing only;
- meaningful end-to-end improvement on the target case.

## Trial 1: detach-before-summary accumulation

Date: 2026-06-02

Hypothesis:

Detached forward-wavefield summaries are not part of the loss gradient. Changing
the summary computation from:

```python
torch.sum(p * p, dim=0).detach()
```

to:

```python
p_summary = p.detach()
torch.sum(p_summary * p_summary, dim=0)
```

could reduce autograd graph work during checkpoint backward replay.

Command before and after:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 3 \
  --checkpoint-segments 10 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/phase_a_profile_detached_summary_tmp
```

Validation:

- `py_compile ADFWI/propagator/acoustic_kernels.py`: passed
- backend integration parity tests for acoustic output policy and pressure-only:
  passed
- loss trajectory matched exactly:
  `6375.7919921875 -> 6006.28125 -> 5718.13330078125`
- raw and processed gradients remained finite

Timing, steady-state average over iterations 2-3:

| Metric | Before | Candidate | Result |
| --- | ---: | ---: | ---: |
| total iteration | 30.2897 s | 30.0972 s | +0.64% |
| forward | 4.4441 s | 4.6420 s | -4.45% |
| backward | 25.1980 s | 24.8069 s | +1.55% |

Decision:

Closed and reverted. The candidate slightly reduced backward time, but the
overall gain was below the useful threshold and forward timing regressed. It is
not a meaningful production optimization.

Next route:

Continue checkpoint/backward work only with changes that reduce recompute or
checkpoint assembly cost more directly. Avoid small expression rewrites unless a
profile shows a specific operator-level cost.

## Trial 2: skip detached illumination summaries during checkpoint replay

Date: 2026-06-02

Hypothesis:

PyTorch reentrant checkpoint runs the original forward under `no_grad` and
replays the segment during backward with gradients enabled. Acoustic
forward-wavefield summaries are detached illumination/visualization outputs and
do not participate in the receiver-loss gradient. They need to be accumulated in
the original forward, but not during checkpoint backward replay.

Implementation:

- added `accumulate_wavefield_in_grad` to the scripted acoustic segment
  functions;
- direct `checkpoint_segments == 1` calls pass `True`;
- checkpointed calls pass `False`, so summaries are accumulated in the original
  `no_grad` forward and skipped in grad-enabled replay;
- receiver outputs, wavefield recurrence, loss inputs, and return structure are
  unchanged.

Validation:

- `py_compile ADFWI/propagator/acoustic_kernels.py`: passed
- backend integration parity tests for acoustic output policy and pressure-only:
  passed
- reduced checkpoint=10 loss trajectory matched exactly:
  `6375.7919921875 -> 6006.28125 -> 5718.13330078125`
- full-record checkpoint=10 loss trajectory matched exactly:
  `74756.2578125 -> 71691.453125 -> 68891.140625`
- raw and processed gradients remained finite
- checkpoint=1 sanity run completed with finite loss and gradients

Timing, steady-state average over iterations 2-3:

| Case | Metric | Before | Candidate | Result |
| --- | --- | ---: | ---: | ---: |
| reduced, checkpoint=10 | total iteration | 30.2897 s | 29.2227 s | +3.52% |
| reduced, checkpoint=10 | forward | 4.4441 s | 4.4721 s | -0.63% |
| reduced, checkpoint=10 | backward | 25.1980 s | 24.1039 s | +4.34% |
| full-record, checkpoint=10 | total iteration | 28.2031 s | 27.4098 s | +2.81% |
| full-record, checkpoint=10 | forward | 4.2238 s | 4.2271 s | -0.08% |
| full-record, checkpoint=10 | backward | 23.3267 s | 22.5318 s | +3.41% |

Decision:

Accepted as a production checkpoint-path optimization. The improvement is
modest but real on both reduced and full-record FWI loops, and it targets the
dominant backward replay cost without changing the scientific outputs.

Next route:

The remaining checkpoint cost is still dominated by full wavefield recompute.
Further work should focus on reducing replayed recurrence cost or checkpoint
assembly overhead, not detached summary expressions.

## Trial 3: use empty placeholders for skipped replay summaries

Date: 2026-06-02

Hypothesis:

After Trial 2, checkpoint backward replay no longer accumulates detached
illumination summaries, but the segment still initializes same-shape
`forward_wavefield_*` outputs with zeros. Replacing those replay-only
placeholders with `torch.empty` could avoid zero-fill cost while preserving
output metadata.

Implementation:

- only initialize `forward_wavefield_*` with zeros when the segment will
  actually accumulate summaries;
- use same-shape `torch.empty` placeholders when summaries are skipped.

Validation:

- `py_compile ADFWI/propagator/acoustic_kernels.py`: passed
- backend integration parity tests for acoustic output policy and pressure-only:
  passed
- reduced checkpoint=10 loss trajectory matched exactly:
  `6375.7919921875 -> 6006.28125 -> 5718.13330078125`
- raw and processed gradients remained finite

Timing, compared against the accepted Trial 2 state, steady-state average over
iterations 2-3:

| Metric | Trial 2 | Candidate | Result |
| --- | ---: | ---: | ---: |
| total iteration | 29.2227 s | 29.8362 s | -2.10% |
| forward | 4.4721 s | 4.4712 s | +0.02% |
| backward | 24.1039 s | 24.7174 s | -2.55% |

Decision:

Closed and reverted. The candidate preserved numerical behavior, but it slowed
the target checkpoint path. Keeping the zero-initialized placeholder is better
for the current NPU/TorchScript execution path.

Next route:

Stop optimizing replay summary placeholders. The remaining useful work must
target recurrence replay or checkpoint segment assembly more directly.
