# 00 - Propagator Optimization Outline

## Goal

Audit `ADFWI/propagator` responsibilities and contracts before making any code
changes. The propagator layer is the numerical forward-modeling boundary between
`ADFWI.model`, `ADFWI.survey`, and `ADFWI.fwi`, so changes here require stricter
validation than survey/model readability work.

This first round is documentation-only.

## Scope

Allowed in this planning stage:

- inspect `ADFWI/propagator`;
- inspect direct callers in `ADFWI/fwi`, `ADFWI/dip`, tests, and validation examples;
- document ownership boundaries;
- document data/shape/device/dtype contracts;
- document risk levels and validation requirements;
- identify bounded next tasks.

Non-scope for this round:

- changing finite-difference formulas;
- changing boundary-condition formulas;
- changing checkpoint behavior;
- changing acoustic/elastic output component names or shapes;
- changing gradient processing behavior;
- migrating examples or notebooks;
- splitting kernel files.

## Current Module Roles

| File | Current role | Notes |
| --- | --- | --- |
| `acoustic_propagator.py` | public acoustic propagator wrapper | validates model/survey types, follows backend, snapshots source/receiver tensors, builds boundary damping, calls acoustic kernel |
| `elastic_propagator.py` | public elastic propagator wrapper | validates model/survey types, follows backend, snapshots source/receiver tensors, builds boundary terms, calls elastic kernel |
| `acoustic_kernels.py` | active acoustic finite-difference kernel | JIT scripted step function, checkpointed time chunks, outputs acoustic records |
| `acoustic_kernels_bs.py` | alternate acoustic boundary-save/checkpoint path | not imported by `AcousticPropagator` main path; contains custom checkpoint/boundary-save logic |
| `elastic_kernels.py` | active elastic finite-difference kernels | large file with FD coefficient helpers and PML/ABL kernels for multiple FD orders |
| `boundary_condition.py` | numpy boundary/damping profile builders | creates PML/Gerjan/SinCos profiles consumed by propagator wrappers |
| `gradient_process.py` | gradient post-processing helpers | NumPy/SciPy and torch-native gradient taper/smoothing/normalization; used by FWI/DIP |
| `__init__.py` | public exports | exposes propagators and gradient processors |

## Key Contracts

### Inputs From Model

Acoustic propagator expects:

- model is an `AbstractModel`;
- `model.forward()` refreshes constrained/derived state before each propagation;
- `model.vp` and `model.rho` are tensors on the propagator backend;
- grid metadata: `ox`, `oz`, `nx`, `nz`, `dx`, `dz`, `nabc`, `abc_type`, `free_surface`.

Elastic propagator expects:

- model is an `AbstractModel`;
- `model.forward()` refreshes elastic moduli and staggered-grid quantities;
- kernel-facing tensors include `lamu`, `lam`, `bx`, `bz`, `CC`;
- same grid and boundary metadata as acoustic, plus valid elastic model state.

### Inputs From Survey

Both propagators expect:

- `Source.get_loc()` normal source shape `(src_num, 2)`;
- `Receiver.get_loc()` shape `(rcv_num, 2)`;
- source and receiver coordinates are grid indices;
- `Source.get_wavelet()` normal source shape `(src_num, nt)`;
- acoustic source wavelet and elastic moment tensor are already prepared by `Source`;
- `Survey.receiver_masks` is stored for downstream FWI masking, not applied by the forward kernels.

### Backend Contract

Both propagators:

- default to the model backend when no explicit `device`/`dtype` is provided;
- convert source/receiver geometry and wavelet data to backend tensors during construction;
- expose `.device` and `.dtype` for FWI/runtime alignment.

### Output Contract

Acoustic forward output is consumed as waveform dict keys used by `SeismicData`
and FWI acoustic paths, currently including acoustic receiver components such
as `p`, `u`, and `w`.

Elastic forward output is consumed by elastic FWI paths and `SeismicData`
elastic parsing, currently including elastic components such as `txx`, `tzz`,
`txz`, `vx`, and `vz`.

The common waveform layout used by validation and survey contracts is
`(shot, time, receiver)`.

## Audit Findings

These observations define future optimization targets, not immediate edits.

1. Public wrappers and kernels are reasonably separated:
   - wrappers own model/survey/backend adaptation;
   - kernels own finite-difference time stepping.
2. Wrapper responsibilities are duplicated between acoustic and elastic:
   - type validation;
   - backend selection;
   - model/survey metadata extraction;
   - source/receiver tensor conversion.
3. Boundary-condition behavior is shared but not uniformly named:
   - acoustic wrapper calls `bc_pml`, `bc_gerjan`, `bc_sincos`;
   - elastic wrapper calls `bc_pml_xz`, `bc_gerjan`, `bc_sincos`;
   - acoustic PML currently calls `bc_pml(..., free_surface=False)` regardless of `self.free_surface`.
4. Kernel files contain heavy numerical logic and must not be cleaned casually:
   - `elastic_kernels.py` is large and formula-sensitive;
   - `acoustic_kernels_bs.py` appears to be an alternate or experimental path and is not imported by the main acoustic wrapper;
   - commented legacy blocks exist in kernel code but removing them should wait for dedicated reference validation.
5. `gradient_process.py` is exported from `ADFWI.propagator`, but semantically it is closer to FWI runtime post-processing than propagation.
   Keep it in place for compatibility unless a separate migration plan exists.

## Risk Levels

| Area | Risk | Why |
| --- | --- | --- |
| docs/comments/import readability | low | no numerical behavior change |
| wrapper type/error messages | low-medium | can affect user-facing exceptions |
| wrapper metadata extraction helpers | medium | can silently change tensor shapes/devices |
| boundary-condition wrapper selection | high | affects all wavefields near boundaries |
| checkpoint segment handling | high | can affect memory, autograd graph, and numerical equivalence |
| acoustic/elastic output dict keys or shape | high | affects `SeismicData`, FWI, examples, saved data |
| finite-difference formulas | very high | requires numerical regression and gradient validation |
| gradient processor behavior | high | affects inversion trajectory, not just forward waveforms |

## Validation Policy

Use the lightest validation that proves the change, but treat propagator-facing
changes as numerical unless proven otherwise.

| Change type | Minimum validation |
| --- | --- |
| docs-only plan | `git diff --check` |
| wrapper docstrings/import cleanup | `py_compile`, backend integration |
| wrapper tensor extraction refactor | focused shape/device/dtype tests, backend integration, Marmousi2 validation `check` |
| boundary-condition helper refactor without formula change | exact array comparison for damping profiles |
| acoustic forward wrapper change | small forward output comparison: shape, dtype, finite, max_abs/max_rel vs baseline |
| elastic forward wrapper change | small elastic forward output comparison across components |
| checkpoint behavior change | checkpoint_segments 1 vs N comparison on same case plus backward/gradient check |
| finite-difference kernel change | dedicated numerical reference case, forward max_abs/max_rel, gradient/backward validation, real-case smoke |
| gradient processor change | NumPy vs torch exact/tolerance comparison and FWI gradient-flow check |

## Recommended Optimization Path

### Phase 1 - Boundary And Contract Documentation

Status: current round.

Goal:
document ownership, contracts, risks, and validation rules.

Validation:
`git diff --check`.

Stop:
no propagator code edits.

### Phase 2 - Wrapper Readability Only

Potential bounded target:

- replace stale file headers in `acoustic_propagator.py`, `elastic_propagator.py`,
  `boundary_condition.py`, and `gradient_process.py` with concise module
  docstrings;
- remove obviously unused imports in wrappers only if `py_compile` and backend
  integration pass;
- do not touch kernels yet.

Validation:

- `conda run -n adfwi python -m py_compile ADFWI/propagator/*.py`;
- `conda run -n adfwi python -m unittest tests/test_backend_integration.py`;
- `git diff --check`.

### Phase 3 - Wrapper Contract Tests

Potential bounded target:

- add focused tests for propagator construction contracts:
  - follows model backend;
  - source/receiver tensors are on the propagator backend;
  - source/receiver tensor shapes match survey contracts;
  - receiver masks are stored but not applied by forward kernels.

Validation:
focused unit tests plus backend integration.

### Phase 4 - Boundary Profile Validation

Potential bounded target:

- test `bc_pml`, `bc_pml_xz`, `bc_gerjan`, and `bc_sincos` output shape and
  finite values for free-surface and non-free-surface cases;
- only after tests exist, consider readability refactors inside
  `boundary_condition.py`.

Validation:
exact or tolerance array comparisons for existing behavior.

### Phase 5 - Kernel Work Only If Needed

Kernel changes should not start from cleanup preference. They should require a
clear bug, performance bottleneck, or numerical validation need.

Required before kernel edits:

- define a reference case;
- save baseline output statistics;
- compare every output component;
- check backward/gradient path when used by FWI;
- record device, dtype, checkpoint_segments, and seed/config.

## First Next Task

The recommended next implementation round is **wrapper readability only**:

```text
Goal:
Clarify propagator wrapper responsibilities without changing numerical behavior.

Scope:
ADFWI/propagator/acoustic_propagator.py
ADFWI/propagator/elastic_propagator.py
ADFWI/propagator/__init__.py only if needed
docs/version-plans/bv1.2-propagator/

Validation:
py_compile, tests/test_backend_integration.py, Marmousi2 validation check only
if constructor-facing behavior changes.

Stop:
No kernel edits, no boundary formula edits, no checkpoint edits.
```

## Stop Rule

Do not continue into kernels, boundary formulas, checkpoint logic, or gradient
processing unless there is a new bounded task with explicit numerical
validation and a stop condition.
