# 17 - Acoustic Pressure Inner-State Probe

Date: 2026-05-31

## Purpose

This Phase B probe tested one final candidate closer to the recurrent acoustic
update graph.

Hypothesis:

```text
If the pressure update keeps only the interior pressure state instead of
writing into a full pressure tensor slice every step, it may reduce
SliceBackward / CopySlices overhead.
```

This was intentionally implemented as an isolated benchmark, not as a
production kernel change.

## Script

```text
scripts/benchmark/acoustic_pressure_inner_state_microbenchmark.py
```

The script compares:

- reference: full pressure tensor with recurrent interior sliced assignment;
- candidate: recurrent `p_inner` state only.

It is configured to compare:

- final pressure interior output;
- scalar loss;
- gradients for `p`, `u_seq`, `w_seq`, `kappa1`, and `alpha1`.

## Command

```bash
timeout 300s conda run -n adfwi python scripts/benchmark/acoustic_pressure_inner_state_microbenchmark.py \
  --device npu:0 \
  --dtype float32 \
  --nt 800 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --repeat 3 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_pressure_inner_state_microbenchmark_20260531.json
```

Syntax check:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_pressure_inner_state_microbenchmark.py
```

## Result

The benchmark did not reach timing comparison. The reference recurrent
full-field sliced assignment failed in backward:

```text
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation: [npuFloatType [1, 67, 136]], which is output 0
of AsStridedBackward0, is at version 800; expected version 799 instead.
```

This failure occurred during warmup before candidate comparison.

## Interpretation

This is a useful negative result.

The production acoustic kernel currently relies on the scripted
`step_forward` path to make the recurrent in-place update executable under
autograd. Pulling this recurrence into an ordinary Python-level state rewrite is
not a small mechanical optimization; it changes the autograd representation of
the core recurrence and immediately hits in-place version constraints.

Because this branch requires exact gradient preservation, this candidate cannot
advance without becoming a larger algorithmic rewrite.

## Decision

Reject this as a default-path optimization task.

Do not rewrite the recurrent `p/u/w` state layout inside
`ADFWI/propagator/acoustic_kernels.py` as part of the current performance pass.
Any future attempt belongs in a separate research branch with:

- an independent adjoint or gradient reference;
- tiny CPU/NPU parity tests;
- full reduced FWI numerical comparison;
- full-record validation before merge.

## Phase B Status

Phase B has now produced:

- one accepted default improvement earlier in the branch
  (`checkpoint_segments == 1` no-checkpoint path);
- one accepted opt-in output-policy improvement
  (`save_forward_wavefield=False` when illumination is disabled);
- multiple rejected default-path candidates:
  - `torch.cat` functional reconstruction;
  - `torch.compile` on the current NPU environment;
  - receiver list stacking in production FWI;
  - pressure inner-state recurrence rewrite.

## Next Direction

Pause acoustic default-path Phase B optimization.

The next productive route should be either:

1. Phase D: profile and optimize FWI-loop / gradient-processing overhead if a
   full iteration profile shows it is worth the effort; or
2. Phase E: start elastic propagator profiling using the same profile-first
   discipline.

Continuing to make small acoustic recurrent-kernel rewrites is likely to
produce more rejected probes or gradient risk rather than a safe default
speedup.
