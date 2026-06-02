# Propagator Performance Test Matrix

Use this matrix before accepting any propagator performance change.

## Required Levels

| Level | Purpose | Required when |
| --- | --- | --- |
| Static | import and syntax safety | every propagator edit |
| Contract | wrapper and option behavior | wrapper/helper/API edits |
| Numerical parity | catch waveform/loss/gradient drift | every kernel or FWI-output edit |
| Reduced workflow | confirm real loop does not break | meaningful performance edits |
| Full-record validation | anchor major decisions | milestone changes only |

## Static And Contract Tests

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/*.py
```

```bash
conda run -n adfwi python -m unittest \
  tests.test_backend_integration \
  tests.test_boundary_conditions \
  tests.test_torch_grad_processor
```

For FWI runtime-facing output changes:

```bash
conda run -n adfwi python -m unittest \
  tests.test_fwi_runtime \
  tests.test_fwi_iteration
```

## Acoustic Parity Requirement

Run for any change to `acoustic_kernels.py`, `acoustic_propagator.py`, or
acoustic FWI output policy.

Record:

| Item | Required record |
| --- | --- |
| device/dtype | e.g. `cpu/float32`, `npu:0/float32` |
| outputs | `p`, `u`, `w`, forward wavefield tensors |
| finiteness | all checked tensors finite |
| forward difference | max abs and max relative difference |
| loss difference | absolute and relative difference |
| gradient difference | max abs and max relative `vp.grad` difference |
| timing | forward/backward/total or seconds per iteration |

Default tolerance for pure refactors:

```text
cpu float32: max_abs <= 1e-6 or max_rel <= 1e-5
npu float32: max_abs <= 1e-5 or max_rel <= 1e-4
```

Exact-code changes should aim for zero difference where the backend allows it.

## Benchmark Commands

Checkpoint/output policy probe:

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

FWI iteration profile:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --device npu:0 \
  --dtype float32 \
  --iterations 5 \
  --checkpoint-segments 1
```

Pressure-only acoustic FWI opt-in comparison:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_fwi_iteration_profile.py \
  --validation-case reduced \
  --device npu:0 \
  --dtype float32 \
  --iterations 2 \
  --checkpoint-segments 10 \
  --pressure-only
```

Required result: `p`, `forward_wavefield_p`, pressure loss, and raw `vp.grad`
match the full-output path. `u/w` receiver outputs and `u/w` wavefield summaries
are explicit zero placeholders in the pressure-only path and must not be used by
callers.

Custom chunk gate:

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

## Reduced And Full-Record Validation

Reduced inversion:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/run_validation.py inversion10 \
  --iterations 10 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Full-record forward:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Full 300-iteration inversion is reserved for major milestones after reduced
tests and full-record forward pass.

## Full-Record Baseline

| Metric | Baseline |
| --- | ---: |
| `record.p.shape` | `[40, 3000, 200]` |
| `record.p.norm` | `4.604166507720947` |
| forward wall time | `7.913247490301728 s` |
| initial loss | `74756.2578125` |
| final loss | `4211.63671875` |
| min loss | `2709.09228515625` |
| `vp_update_norm` | `43990.0078125` |
| seconds / iteration | `33.75309997430071 s` |

## Acceptance Rule

A performance change is acceptable only if:

- the same command, device, dtype, and case are used before and after;
- waveform/loss/gradient differences are reported;
- timing improves enough to matter for the target case;
- the result is added to `performance-change-log.md`;
- the next direction is explicitly stated.
