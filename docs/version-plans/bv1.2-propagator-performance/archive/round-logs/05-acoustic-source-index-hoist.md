# Acoustic Source Index Hoist

Date: 2026-05-31

## Boundary

```text
Goal:
  Remove a repeated small tensor allocation from the acoustic timestep hot loop.

Scope:
  ADFWI/propagator/acoustic_kernels.py only for code.

Validation:
  Compile, checkpoint/direct numerical parity, gradient parity, reduced
  validation, full-record forward comparison, and existing propagator/backend
  contract tests.

Stop:
  Stop after source batch index hoisting is validated. Do not change acoustic
  finite-difference updates, output keys, receiver sampling, checkpoint policy,
  or elastic kernels in this round.
```

## Code Change

File:

```text
ADFWI/propagator/acoustic_kernels.py
```

Change:

```text
forward_kernel:
  src_index = torch.arange(src_n, dtype=torch.long, device=device)

step_forward:
  use p[src_index, src_z, src_x] for source injection
```

Before this change, `torch.arange(src_n)` was constructed twice inside every
time step for standard, non-encoded acoustic source injection. The new path
builds the index tensor once in `forward_kernel` and passes it to the scripted
`step_forward` function. The source update, receiver recording, forward
wavefield accumulation, checkpoint behavior, and output dictionaries are
unchanged.

## Numerical Parity

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_checkpoint_overhead.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --checkpoint-segments 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --dx 40 \
  --dz 40 \
  --dt 0.003 \
  --f0 5 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_source_index_after_20260531.json
```

Result:

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

The benchmark still compares the implemented `checkpoint_segments == 1` path
against the direct no-checkpoint monkeypatch. Output and `vp` gradient parity
are exactly equal for this controlled case.

## Timing

Warm NPU timing after the source-index hoist:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| implemented checkpoint_segments=1 path | 1.6549 | 4.8917 | 6.5466 |
| benchmark direct monkeypatch | 1.6311 | 4.8512 | 6.4824 |

Previous warm timing after checkpoint bypass, before this source-index hoist
from `03-acoustic-checkpoint-bypass-implementation.md`:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| implemented checkpoint_segments=1 path | 1.8277 | 5.0357 | 6.8634 |

Approximate change on the small NPU benchmark:

| Metric | Before | After | Change |
| --- | ---: | ---: | ---: |
| forward mean | 1.8277 s | 1.6549 s | about 9.5% lower |
| backward mean | 5.0357 s | 4.8917 s | about 2.9% lower |
| total mean | 6.8634 s | 6.5466 s | about 4.6% lower |

These numbers are a hot-path indicator. They are not a full 300-iteration FWI
speedup estimate.

## Reduced Validation

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 2 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/source_index_validation
```

Result:

```text
status: ok
forward record.p.shape: [3, 3000, 200]
forward record.p.norm: 1.3137102127075195
forward seconds: 7.362000536173582
inversion iterations: 2
initial_loss: 6375.7919921875
final_loss: 6006.28125
vp_update_norm: 2295.9599609375
inversion seconds: 52.596621464937925
```

The reduced inversion loss decreases over the two checked iterations.

## Full-Record Forward Comparison

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_full_record/outputs/source_index_full_forward
```

Result:

```text
status: ok
record.p.shape: [40, 3000, 200]
record.p.dtype: float32
record.p.finite: true
record.p.min: -0.04947379231452942
record.p.max: 0.09417726844549179
record.p.norm: 4.604166507720947
seconds: 6.982449723407626
```

Baseline from `full-record-marmousi2-baseline.md`:

```text
record.p.norm: 4.604166507720947
seconds: 7.913247490301728
```

Forward numerical summary matches the full-record baseline.

## Unit/Static Tests

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/acoustic_kernels.py
```

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

Result:

```text
Ran 35 tests in 11.451s
OK
```

## Remaining Risk

- Full 300-iteration FWI was not rerun for this small hot-path change.
- Encoded-source injection still uses the original loop branch and was not
  optimized in this round.
- Timing can vary between NPU runs; the reliable correctness signal is the zero
  output and gradient difference in the controlled benchmark plus the
  full-record forward norm match.

## Next Direction

Use profiling to decide whether the next acoustic hot path should target
receiver sampling/output accumulation or forward-wavefield accumulation. Do
not start another structural cleanup unless it is tied to a measured timing
cost and a clear numerical comparison.
