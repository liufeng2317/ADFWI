# 16 - Acoustic Receiver Recording Stack Probe

Date: 2026-05-31

## Purpose

This Phase B round tested whether the acoustic receiver-output recording path
is a useful default optimization target.

Hypothesis:

```text
Replacing per-time-step preallocated receiver writes
    rcv[:, it, :] = sampled_wavefield
with list collection plus torch.stack
could reduce CopySlices / SliceBackward overhead while preserving gradients.
```

The production kernel was changed temporarily only after the isolated benchmark
showed exact gradient parity. The production change was then rejected and
reverted after the reduced FWI validation showed no end-to-end speedup.

## Isolated Benchmark

Script:

```text
scripts/benchmark/acoustic_receiver_recording_microbenchmark.py
```

Command:

```bash
timeout 240s conda run -n adfwi python scripts/benchmark/acoustic_receiver_recording_microbenchmark.py \
  --device npu:0 \
  --dtype float32 \
  --nt 800 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --receivers 100 \
  --repeat 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_receiver_recording_microbenchmark_20260531.json
```

Result:

| Metric | Result |
| --- | ---: |
| loss max abs diff | `0.0` |
| p/u/w record max abs diff | `0.0` |
| p/u/w sequence grad max abs diff | `0.0` |
| forward speedup mean | `2.3997x` |
| backward speedup mean | `1.6279x` |
| total speedup mean | `1.7669x` |

Interpretation:

The receiver-recording pattern by itself benefits from list collection plus
`torch.stack`, and it preserves gradients exactly in the isolated setup.

## Production Parity Check

The same idea was temporarily applied to `ADFWI/propagator/acoustic_kernels.py`
inside `step_forward`.

Small acoustic forward/backward parity was checked against a detached `HEAD`
worktree at commit `9ac6155`.

Compared tensors:

- `p`, `u`, `w`;
- `forward_wavefield_p`, `forward_wavefield_u`, `forward_wavefield_w`;
- scalar loss;
- raw `vp.grad`.

Result:

| Metric | Result |
| --- | ---: |
| loss abs diff | `0.0` |
| all record and forward-wavefield max abs diff | `0.0` |
| `vp.grad` max abs diff | `0.0` |

This confirmed that the candidate did not change numerical behavior in the
tested production path.

## Reduced FWI Comparison

The candidate was then tested in the reduced Marmousi2 FWI iteration profile:

```bash
timeout 600s conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --shots 3 \
  --checkpoint-segments 1 \
  --iterations 1 \
  --generate-observed \
  --dataset-dir /liufeng1afs/project/04_Inversion/ADFWI-github/examples/datasets/marmousi2_source \
  --output-root /tmp/adfwi_current_receiver_stack_profile \
  --result-json /tmp/adfwi_current_receiver_stack_profile.json
```

Reference was run from a detached worktree at commit `9ac6155` with the same
arguments.

| Metric | Reference | Candidate | Ratio |
| --- | ---: | ---: | ---: |
| loss | `6375.7919921875` | `6375.7919921875` | exact |
| raw grad norm | `0.8779116272926331` | `0.8779116272926331` | exact |
| processed grad norm | `67952.5546875` | `67952.5546875` | exact |
| total time | `25.6777 s` | `26.7458 s` | `0.9601x` |
| forward time | `7.0406 s` | `7.0253 s` | `1.0022x` |
| backward time | `17.2309 s` | `18.2817 s` | `0.9425x` |

## Decision

Reject the production change.

Reason:

- numerical parity is exact, but the reduced FWI iteration is slower;
- the isolated receiver-recording speedup does not transfer to the real
  acoustic kernel because the full backward path is dominated by the recurrent
  wavefield update graph, not receiver-output writes alone.

The production `ADFWI/propagator/acoustic_kernels.py` change was reverted.
Only the benchmark and this record are kept.

## Next Direction

Return to the main Phase B route.

Receiver recording is not the next default-path optimization. The next viable
target should be closer to the recurrent `p/u/w` update graph itself, or Phase B
should be paused after one more bounded probe if the next candidate also fails
to show real reduced-FWI speedup.
