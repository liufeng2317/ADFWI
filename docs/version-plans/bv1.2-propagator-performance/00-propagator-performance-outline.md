# Propagator Performance Optimization Outline

Date: 2026-05-30

## Boundary

```text
Goal:
  Build the performance optimization route for ADFWI/propagator.

Scope:
  ADFWI/propagator analysis and documentation only.

Validation:
  Static inspection, py_compile, and existing lightweight propagator/backend
  tests. No kernel numerical behavior changes in this round.

Stop:
  Stop after the route and test matrix are recorded. Do not rewrite kernels in
  the planning round.
```

## Current Propagator Structure

`ADFWI/propagator` currently has four functional layers.

| Layer | Files | Role |
| --- | --- | --- |
| Public wrapper | `acoustic_propagator.py`, `elastic_propagator.py` | Bind model, survey, backend/device/dtype, boundary arrays, source/receiver tensors, and call kernels. |
| Boundary setup | `boundary_condition.py` | Build PML, sine/cosine, Gerjan, or combined x/z damping arrays. This is construction-time setup, not the timestep hot loop. |
| Forward kernels | `acoustic_kernels.py`, `elastic_kernels.py` | Main wave propagation operators. These own tensor padding, timestep loops, checkpoint segmentation, receiver recording, and forward-wavefield accumulation. |
| Gradient post-process | `gradient_process.py` | FWI gradient masking/smoothing/normalization. It is performance relevant in inversion, but it is not the propagator forward kernel. |

`acoustic_kernels_bs.py` is an experimental boundary-saving implementation. It
is not imported by the public wrappers and must not be used as the first
baseline unless it is explicitly promoted behind a tested option.

## Hot Paths

Acoustic forward path:

```text
AcousticPropagator.forward
  -> acoustic_kernels.forward_kernel
     -> pad_torchSingle(v/rho)
     -> allocate p/u/w, receiver records, forward wavefields
     -> for each checkpoint segment
        -> checkpoint(step_forward)
           -> for each timestep
              -> update p/u/w
              -> inject source
              -> sample receivers
              -> accumulate forward_wavefield_p/u/w
```

Elastic forward path:

```text
ElasticPropagator.forward
  -> elastic_kernels.forward_kernel
     -> pad lamu/lam/bx/bz/Cij and boundary arrays
     -> allocate stress/velocity state, receiver records, forward wavefields
     -> choose PML or ABL branch
     -> choose fd_order 4/6/8/10 step function
     -> checkpoint(step_forward_...)
        -> for each timestep
           -> finite-difference stress/velocity update
           -> inject moment tensor source
           -> free-surface and boundary handling
           -> sample receivers
           -> accumulate five forward wavefields
```

The real compute cost is inside the timestep loops. Wrapper cleanup, import
cleanup, and boundary setup are useful for readability but are not expected to
move the full-record performance baseline.

## Baseline To Preserve

Use `docs/version-plans/bv1.2/full-record-marmousi2-baseline.md` as the first
performance baseline:

| Metric | Value |
| --- | --- |
| Case | `examples/validation/marmousi2_acoustic_full_record` |
| Device / dtype | `npu:0`, `float32` |
| `checkpoint_segments` | `1` |
| Forward shape | `[40, 3000, 200]` |
| Forward pressure norm | `4.604166507720947` |
| Forward wall time | `7.913247490301728 s` |
| 300-iter inversion final loss | `4211.63671875` |
| 300-iter inversion min loss | `2709.09228515625` at iter 283 |
| 300-iter inversion time | `10125.929992290214 s` |
| Seconds / iteration | `33.75309997430071 s` |

Any future kernel-level optimization must compare both timing and numerical
results against this baseline or a newly documented pre-change baseline from
the same command and hardware.

## Optimization Route

This route is profile-first. Static code inspection is useful for understanding
the operator, but it is not enough to choose the next optimization. Every
kernel-level change must start from a measured bottleneck on a named case.

### Phase 1: Measurement Harness

Purpose: make before/after measurement reproducible before editing kernels and
identify the dominant cost before deciding what to optimize.

Tasks:

- add a small propagator benchmark command or script around existing validation
  scripts;
- record wall time, device, dtype, `checkpoint_segments`, shot count, receiver
  count, `nt`, and output summary;
- produce comparable JSON summaries for forward and short inversion runs.
- separate at least forward, backward, and gradient-processing time for a
  differentiable acoustic run;
- separate one-time setup cost from `propagator.forward` for full-record
  forward runs.

Required validation:

- `py_compile` for `ADFWI/propagator`;
- backend/propagator unit tests;
- reduced acoustic validation forward;
- full-record forward only when comparing real timing.

### Phase 2: Acoustic Backward/Forward Hot-Path Work

Purpose: optimize the measured hot path without changing public behavior. The
first probe shows backward dominates the differentiable acoustic run, so
backward-safe changes have priority over gradient post-processing cleanup.

Candidate targets:

- completed: direct `step_forward` call when `checkpoint_segments=1` for the
  acoustic kernel; see `03-acoustic-checkpoint-bypass-implementation.md`;
- completed: hoist invariant source index tensors out of the timestep loop;
  see `05-acoustic-source-index-hoist.md`;
- avoid repeated small allocations where the same tensor can be safely reused;
- remove unused imports only if they do not trigger JIT or runtime side effects.
- inspect whether forward-wavefield accumulation and recording extra components
  affect backward graph cost.

Required validation:

- acoustic tiny numerical parity: `p`, `u`, `w`, and forward wavefield max
  absolute/relative differences;
- gradient parity on a small differentiable acoustic case;
- reduced validation forward and short inversion;
- full-record forward timing before/after.

### Phase 3: Output Policy Exploration

Purpose: separate required FWI records from visualization-only wavefield
summaries if profiling proves output accumulation is material.

Candidate targets:

- optional recording policy for pressure-only acoustic runs;
- completed: opt-in acoustic forward-wavefield accumulation control;
  see `07-acoustic-forward-wavefield-policy.md`;
- avoid computing unused receiver components only behind an explicit option.

Risk:

- this touches public output dictionaries and FWI assumptions. It must be
  opt-in or maintain the default output exactly.
- acoustic FWI uses `forward_wavefield_p` for GradProcessor illumination
  preconditioning; skipped forward wavefields must not be enabled when
  `forw_illumination=True`.

Required validation:

- default mode bitwise or near-bitwise parity;
- opt-in mode shape and key-contract tests;
- FWI short inversion verifies the loss path still uses the intended component.

### Phase 4: Elastic Profiling After Acoustic Baseline

Purpose: only optimize elastic kernels after the acoustic performance workflow
is proven.

Candidate targets:

- profile PML vs ABL branches separately;
- profile `fd_order` 4/6/8/10 separately;
- inspect duplicated step functions only after hot operators are measured;
- avoid structural consolidation unless it improves performance or reduces a
  verified correctness risk.

Required validation:

- elastic tiny forward parity for `txx`, `tzz`, `txz`, `vx`, `vz`;
- `tests/test_backend_integration.py` elastic cases;
- representative elastic example smoke run if the kernel changes.

### Phase 5: Gradient Processing In Inversion

Purpose: revisit `GradProcessor` only if full FWI profiling shows it is visible.
The first acoustic bottleneck probe shows gradient post-processing is not the
dominant cost for the measured case.

Candidate targets:

- profile NumPy/SciPy gradient post-processing in the inversion loop;
- compare `GradProcessor` and `TorchGradProcessor` on masks, smoothing, and
  normalization;
- avoid changing default behavior until parity is documented for the target
  workflow.

Required validation:

- `tests/test_torch_grad_processor.py`;
- short inversion numerical comparison;
- full-record timing only after short parity passes.

## Do Not Start With

- rewriting elastic kernels before acoustic profiling;
- changing finite-difference equations without a reference comparison;
- replacing checkpoint behavior without gradient comparison;
- changing public output keys in default mode;
- optimizing `boundary_condition.py` as a first target, because it is not the
  timestep hot loop;
- promoting `acoustic_kernels_bs.py` without an explicit experiment branch and
  strict numerical comparison.
