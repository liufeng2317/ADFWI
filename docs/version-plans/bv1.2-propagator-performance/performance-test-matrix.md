# Propagator Performance Test Matrix

Use this matrix before accepting any propagator performance change.

## Test Levels

| Level | Purpose | When to run |
| --- | --- | --- |
| Static | Import/compile safety | Every propagator edit |
| Unit contract | Wrapper/backend/boundary/gradient contracts | Every wrapper or helper edit |
| Tiny numerical parity | Catch silent equation/output/gradient drift | Every kernel edit |
| Reduced validation | Verify real workflow still runs | Every meaningful performance edit |
| Full-record forward | Compare real forward timing | Performance milestones |
| Full-record inversion | Compare full FWI behavior | Major milestones only |

Before selecting an optimization target, run at least one profile probe that
separates forward, backward, and gradient-processing time. Do not infer the
first target from static inspection alone.

For acoustic output-side cost probes, use:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_output_cost.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --shots 1 \
  --receivers 200 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --requires-grad
```

This benchmark is only for bottleneck selection. It does not replace acoustic
numerical parity tests because it does not run the full propagator.

For acoustic forward-wavefield output-policy changes, use:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_wavefield_policy.py \
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
  --f0 5
```

Required result: `p/u/w`, pressure loss, and raw `vp.grad` differences are
zero or tolerance-bounded. This does not prove FWI processed-gradient parity
when `GradProcessor.forw_illumination=True`.

For AcousticFWI-level use of `save_forward_wavefield=False`, require guard and
processed-gradient tests:

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_fwi_rejects_skipped_forward_wavefield_when_illumination_is_active \
  tests.test_backend_integration.BackendIntegrationTests.test_acoustic_fwi_skipped_forward_wavefield_matches_default_without_illumination
```

The opt-in FWI path is valid only when active gradient processors explicitly set
`forw_illumination=False`.

For benchmark-only custom backward prototypes, use the chunk-level probe before
considering production integration:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_chunk_forward.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --steps 300 \
  --shots 1 \
  --nx 64 \
  --nz 32 \
  --nabc 16 \
  --receivers 32 \
  --loss-kind receiver-random-linear \
  --loss-components rcv_p \
  --upstream-scale 4500
```

Pair it with a CPU float64 formula gate using the same shape before any
production-facing custom backward change. Required result: exact outputs/loss,
CPU float64 gradients close to machine precision, and documented NPU float32
gradient difference with max absolute and relative errors.

## Static And Unit Commands

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/*.py
```

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

Use additional FWI runtime tests when the edit touches propagator outputs used
inside inversion:

```bash
conda run -n adfwi python -m unittest \
  tests.test_fwi_runtime \
  tests.test_fwi_iteration
```

## Acoustic Numerical Parity

Run for any change to `acoustic_kernels.py` or acoustic wrapper semantics.

Required comparison:

| Item | Required record |
| --- | --- |
| Device/dtype | e.g. `cpu/float32`, `npu:0/float32` |
| Shape | `p`, `u`, `w`, forward wavefields |
| Finiteness | all returned tensors finite |
| Forward difference | max abs and max relative difference for `p`, `u`, `w` |
| Wavefield difference | max abs and max relative difference for all forward wavefields |
| Gradient difference | max abs and max relative difference for model gradient on a tiny case |

Default tolerance for first pass:

```text
cpu float32 exact-code refactor: max_abs <= 1e-6 or max_rel <= 1e-5
npu float32 exact-code refactor: max_abs <= 1e-5 or max_rel <= 1e-4
```

If a change intentionally alters numerical behavior, it is not a pure
performance optimization and needs a separate scientific justification.

For `checkpoint_segments=1` checkpoint bypass work, also run:

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

Required result: zero or tolerance-bounded differences for all acoustic output
tensors, loss, and `vp.grad`; report forward/backward/total timing.

## Elastic Numerical Parity

Run for any change to `elastic_kernels.py` or elastic wrapper semantics.

Matrix:

| Branch | fd_order | Components |
| --- | --- | --- |
| PML | 4 | `txx`, `tzz`, `txz`, `vx`, `vz` |
| ABL | 4 | `txx`, `tzz`, `txz`, `vx`, `vz` |
| PML | 6/8/10 | Run when that order is changed |
| ABL | 6/8/10 | Run when that order is changed |

Record the same shape, finiteness, max abs error, max relative error, and
gradient comparison fields as acoustic.

## Reduced Workflow Validation

Use reduced validation after small performance edits:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

For inversion-path changes:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py inversion10 \
  --iterations 10 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Record:

- command;
- output summary path;
- initial/final loss for inversion;
- wall time and seconds per iteration;
- max numerical difference from the pre-change run when available.

## Full-Record Performance Comparison

Forward command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Compare against the bv1.2 baseline:

| Metric | Baseline |
| --- | --- |
| `record.p.shape` | `[40, 3000, 200]` |
| `record.p.norm` | `4.604166507720947` |
| wall time | `7.913247490301728 s` |

Full 300-iteration inversion is expensive. Run it only after reduced tests and
full-record forward both pass.

Inversion baseline:

| Metric | Baseline |
| --- | --- |
| initial loss | `74756.2578125` |
| final loss | `4211.63671875` |
| min loss | `2709.09228515625` |
| `vp_update_norm` | `43990.0078125` |
| seconds / iteration | `33.75309997430071 s` |

## Acceptance Rule

A propagator performance change is acceptable only when:

- the same command, device, dtype, and case are used before and after;
- numerical differences are reported and within the planned tolerance;
- timing improves enough to matter for the target case;
- the optimization path and next direction are recorded in this folder;
- the commit message names the bounded performance target.
