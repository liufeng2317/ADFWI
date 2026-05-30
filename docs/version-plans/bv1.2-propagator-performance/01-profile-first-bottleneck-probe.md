# Profile-First Bottleneck Probe

Date: 2026-05-30

## Boundary

```text
Goal:
  Measure the current acoustic propagator performance bottleneck before
  choosing the first optimization target.

Scope:
  Measurement only. No propagator kernel or FWI code changes.

Validation:
  Run existing benchmark/validation code on NPU and record timing plus basic
  numerical summaries.

Stop:
  Stop after identifying the first bottleneck and updating the route.
```

## Commands

Small differentiable acoustic probe:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_backend_benchmark.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --gradient-processors none,legacy,torch \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_bottleneck_probe_20260530.json
```

Full-record forward compute probe, without writing figures or observed data:

```text
device: npu:0
dtype: float32
shots: 40
receivers: 200
nt: 3000
checkpoint_segments: 1
```

The full-record command used the validation script builders directly and timed
import/backend/model/survey/propagator construction separately from
`propagator.forward`.

## Results

Small differentiable acoustic probe:

| Gradient processor | Forward (s) | Backward (s) | Gradient process (s) | Total (s) |
| --- | ---: | ---: | ---: | ---: |
| none | 2.3826 | 7.9512 | 0.00003 | 10.3338 |
| legacy | 1.4083 | 7.5438 | 0.00114 | 8.9532 |
| torch | 1.4190 | 7.6236 | 0.01911 | 9.0618 |

Numerical check in the benchmark:

```text
pressure_shape: [1, 800, 3]
pressure_dtype: float32
pressure_device: npu:0
legacy-vs-torch loss max_abs_diff: 0.0
legacy-vs-torch vp_grad_norm max_abs_diff: 0.0
legacy-vs-torch pressure_l2 max_abs_diff: 0.0
```

Full-record forward compute probe:

| Stage | Seconds |
| --- | ---: |
| import runtime modules | 21.0271 |
| configure backend | 0.0005 |
| build true model | 32.6327 |
| build survey | 0.0020 |
| build propagator | 0.1292 |
| `propagator.forward` | 7.7090 |

Forward output summary:

```text
record.p.shape: [40, 3000, 200]
record.p.dtype: float32
record.p.device: npu:0
record.p.min: -0.04947379231452942
record.p.max: 0.09417726844549179
record.p.norm: 4.630015850067139
```

The direct tensor norm is not used as the official numerical baseline. The
official saved full-record forward baseline remains:

```text
docs/version-plans/bv1.2/full-record-marmousi2-baseline.md
record.p.norm: 4.604166507720947
forward wall time: 7.913247490301728 s
```

Use the saved validation baseline for numerical pass/fail. Use the compute
probe to understand where time is spent.

## Interpretation

The first measured bottleneck is the differentiable acoustic forward/backward
path, especially backward:

- backward takes roughly 5.3x the legacy-run forward time in the small probe;
- gradient post-processing is not the first bottleneck in this probe;
- full-record pure `propagator.forward` is close to the existing full-record
  forward wall-time baseline, so the validation baseline is consistent for
  forward timing;
- import and model construction can be large one-time costs, but they are not
  the per-iteration FWI kernel bottleneck.

## Updated Optimization Priority

1. Keep Phase 1 focused on reproducible profiling, not code changes.
2. First kernel optimization should target acoustic backward/forward graph cost.
3. Checkpoint behavior with `checkpoint_segments=1` is a likely first measured
   candidate, but it must pass gradient parity.
4. Do not start with `GradProcessor`; current timing does not justify it as the
   first propagator-performance target.
5. Do not start with boundary setup; it is outside the repeated timestep loop.

## Next Direction

Run a bounded checkpoint-overhead experiment for acoustic propagation:

- compare current `checkpoint(step_forward)` behavior against a controlled
  no-checkpoint path for `checkpoint_segments=1`;
- use a tiny acoustic parity case for `p/u/w`, forward wavefields, loss, and
  `vp` gradient;
- only after parity passes, run reduced validation and full-record forward
  timing.

