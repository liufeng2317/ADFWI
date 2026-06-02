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
