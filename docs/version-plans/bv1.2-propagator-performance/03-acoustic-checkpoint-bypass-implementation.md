# Acoustic Checkpoint Bypass Implementation

Date: 2026-05-30

## Boundary

```text
Goal:
  Implement the acoustic no-checkpoint path when checkpoint_segments == 1.

Scope:
  ADFWI/propagator/acoustic_kernels.py only for code.

Validation:
  Compile, checkpoint-overhead parity benchmark, reduced validation forward
  and 10-iteration inversion, full-record forward comparison, and existing
  propagator/backend contract tests.

Stop:
  Stop after acoustic checkpoint_segments=1 is validated. Do not change elastic
  kernels or output policies in this round.
```

## Code Change

File:

```text
ADFWI/propagator/acoustic_kernels.py
```

Change:

```text
if checkpoint_segments == 1:
    call step_forward directly
else:
    keep torch.utils.checkpoint.checkpoint(step_forward, ...)
```

The finite-difference update, source injection, receiver sampling, output keys,
and forward-wavefield accumulation are unchanged.

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
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_checkpoint_bypass_after_warm_20260530.json
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

After implementation, the benchmark's `checkpoint` mode already uses the new
`checkpoint_segments == 1` direct path. The `direct` monkeypatch remains a
sanity check that the implemented path matches a direct `step_forward` call.

Warm repeat timing after implementation:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| implemented checkpoint_segments=1 path | 1.8277 | 5.0357 | 6.8634 |
| benchmark direct monkeypatch | 1.8105 | 5.0480 | 6.8586 |

The two paths are effectively equivalent after the implementation, which is the
expected result.

Compared with the pre-implementation checkpoint-overhead experiment:

| Metric | Before checkpoint path | After implemented path | Change |
| --- | ---: | ---: | ---: |
| backward mean | 7.6549 s | 5.0357 s | about 34% lower |
| total mean | 9.3957 s | 6.8634 s | about 27% lower |

These numbers are from the small NPU benchmark and should be interpreted as a
hot-path indicator, not as full FWI iteration speedup.

## Reduced Validation

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py all \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --iterations 10 \
  --output-root examples/validation/marmousi2_acoustic_reduced/outputs/checkpoint_bypass_validation
```

Result:

```text
status: ok
shots: 3
receivers: 200
nt: 3000
forward record.p.shape: [3, 3000, 200]
forward record.p.norm: 1.3137102127075195
forward seconds: 7.846681334078312
inversion iterations: 10
initial_loss: 6375.7919921875
final_loss: 4776.75927734375
vp_update_norm: 8410.6181640625
inversion seconds: 276.6070708986372
```

The reduced inversion loss decreases monotonically over the 10 iterations.

## Full-Record Forward Comparison

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_full_record/outputs/checkpoint_bypass_full_forward
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
seconds: 7.839245941489935
```

Baseline from `full-record-marmousi2-baseline.md`:

```text
record.p.norm: 4.604166507720947
seconds: 7.913247490301728
```

Forward numerical summary matches the baseline. Full-record forward timing is
slightly lower, but this change mainly targets differentiable backward and FWI
iteration cost.

## Unit/Static Tests

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/acoustic_kernels.py scripts/benchmark/acoustic_checkpoint_overhead.py
```

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

Result:

```text
Ran 35 tests in 11.276s
OK
```

## Remaining Risk

- Full 300-iteration FWI timing was not rerun in this round.
- Direct no-checkpoint mode can use more autograd graph memory than checkpoint
  mode; current NPU full-record memory is sufficient for `checkpoint_segments=1`.
- Elastic kernels still use the original checkpoint behavior.

## Next Direction

Run a short full-record inversion timing comparison, for example 10 iterations
with the existing full-record validation script, before extending the same
strategy to elastic kernels or other checkpoint segment settings.

