# bv1.2 Development Plan

## Positioning

`bv1.2` is the next active development branch after `bv1.1`.

The main goal is not to add many isolated features immediately. Instead, `bv1.2`
should make ADFWI easier to extend, test, maintain, and benchmark while keeping
the existing scientific workflow compatible as much as possible.

## Baseline

- Stable branch: `bv1.1`
- Frozen anchor: `v1.1-freeze`
- Development branch: `bv1.2`

## Main Goals

1. Clarify the framework architecture.
2. Separate reusable core logic from experiment-specific scripts.
3. Improve extensibility for ADFWI forward modeling, backpropagation, propagators,
   misfits, regularization methods, and optimizers.
4. Add a minimal test suite to protect the main workflow.
5. Introduce a unified backend and device-dispatch layer so ADFWI can use CPU, CUDA GPU, or NPU through one framework-level setting.
6. Prepare benchmark tools before doing performance-oriented changes.
7. Keep old examples available while introducing cleaner examples for new usage.

## Non-Goals

1. Do not rewrite the finite-difference propagator kernels as the first step.
2. Do not remove existing examples during early `bv1.2` development.
3. Do not break the current acoustic, elastic, and VTI workflows without a
   compatibility path.
4. Do not mix large generated figures or notebook execution outputs with core
   framework changes unless they are intentionally part of documentation.
5. Do not make NPU compatibility depend on scattered device-specific branches in
   model, propagator, kernel, and FWI loop code. Prefer a small backend abstraction layer.
6. Do not include DR-FWI, DIP, or neural-network reparameterization modules in the
   first NPU compatibility phase.

## Proposed Architecture Work

### 1. Core API Definition

Define and document the responsibility of the main components:

- `Model`: parameterization, bounds, masks, and physical parameter conversion.
- `Survey` and `SeismicData`: acquisition geometry and waveform data storage.
- `Propagator`: forward simulation and standard waveform output.
- `Misfit`: loss calculation between synthetic and observed data.
- `Regularization`: penalty terms for model parameters.
- `FWI Engine`: inversion loop, batching, optimizer step, scheduler step,
  gradient processing, caching, and logging.

Expected output:

- `docs/version-plans` keeps high-level plans.
- A future `docs/architecture` or `ARCHITECTURE.md` records stable API contracts.

### 2. Device Backend Abstraction

Start from a `backends` module, but design it as the foundation for unified ADFWI device scheduling. The goal is to let users configure the device once at the framework level, while models, propagators, kernels, FWI loops, misfits, regularization methods, and benchmarks inherit the same backend consistently.

Candidate responsibilities:

- resolve requested device strings such as `cpu`, `cuda`, `cuda:0`, `npu`, and
  `npu:0` into a backend-aware device object or canonical string;
- detect available backend modules, including optional NPU runtime support;
- provide helpers for tensor creation, dtype control, and `.to(device)` movement;
- expose a global or scoped ADFWI device context for one-line configuration;
- allow explicit per-object override only when needed, while keeping framework defaults simple;
- centralize random seed setup and synchronization hooks;
- report backend name, device count, device capability, and unsupported features;
- provide safe CPU fallback for tests that do not require accelerator kernels.

Initial target:

```python
import ADFWI

ADFWI.set_backend("npu:0")
backend = ADFWI.backend()
device = backend.device
backend.synchronize()
```

Lower-level backend functions remain available from `ADFWI.backends` for tests and migration code:

```python
from ADFWI.backends import configure_backend, get_backend, use_backend
```

A more advanced scoped form can be considered later:

```python
from ADFWI.backends import use_backend

with use_backend("cuda:0"):
    # model, propagator, and FWI engine use the scoped backend by default
    ...
```

User-facing design target:

- simple scripts should need only one backend configuration call;
- examples should avoid repeating `device=device` in every object once the
  framework default is configured;
- backend diagnostics should make it obvious whether the run is using CPU, CUDA
  GPU, NPU, or a fallback path;
- existing explicit `device=` arguments should remain supported during migration.

The first NPU-compatible path should be a small acoustic forward smoke test. More
complex elastic, VTI, and optimizer workflows can be validated after the
core backend layer is stable.


### 3. Data Transform Pipeline

Move waveform processing out of the FWI loop where possible.

Candidate transforms:

- receiver mask
- data mask
- offset mute
- first-arrival mute
- low-pass filtering
- trace normalization

Current bv1.2 status:

- pure torch `TraceNormalize`, `ReceiverMask`, and `DataMask` transforms exist with CPU/NPU tests;
- `AcousticFWI` routes default waveform normalization through transforms;
- `AcousticFWI` now routes synthetic-side `data_masks` through `DataMask(required=False, apply_to="synthetic")`;
- `ElasticFWI` now uses the same transform entry for default normalization and synthetic-side sample masks across pressure, vx, and vz components;
- receiver-mask migration, mute windows, and low-pass filtering remain in the legacy FWI path until separate shape/device validation is complete.

Target style:

```python
DataTransformPipeline([
    OffsetMute(...),
    FirstArrivalMute(...),
    LowPassFilter(...),
    TraceNormalize(...),
])
```

### 4. Shared FWI Engine

Reduce repeated logic across:

- `AcousticFWI`
- `ElasticFWI`
- future ADFWI inversion classes

The shared engine should handle:

- shot batching
- closure and non-closure optimizers
- data transform application
- data misfit calculation
- regularization calculation
- gradient processing
- result caching
- logging and progress reporting

### 5. Testing

Add a small `tests/` suite before large refactors.

Initial test candidates:

- acoustic homogeneous forward smoke test
- backend resolution tests for `cpu`, `cuda` when available, and `npu` when available
- NPU acoustic forward smoke test on the target NPU system
- model shape and bound checks
- `Misfit_L2` backward/gradient smoke test
- receiver mask and data mask shape behavior
- one tiny inversion loop with 1 or 2 iterations

### 6. Configuration-Driven Examples

Keep existing examples, but start adding new examples using configuration files.

Candidate structure:

```text
examples/bv1.2_quickstart/
  acoustic_forward.yaml
  acoustic_inversion.yaml
  run_example.py
```

This should reduce duplicated example scripts over time.

### 7. Benchmark Preparation

Create benchmark scripts before performance optimization.

Metrics to record:

- forward runtime
- inversion iteration runtime
- accelerator memory usage for CUDA GPU or NPU
- loss curve
- gradient norm
- final model error metrics
- backend metadata, including backend name, device index, dtype, and fallback status

## Suggested Milestones

### Milestone 1: Documentation, Backend Layer, and Safety Net

- Add version plans.
- Add architecture draft.
- Add unified backend and device-dispatch abstraction for CPU, CUDA GPU, and NPU.
- Add minimal tests, including backend resolution and NPU acoustic smoke tests.
- Confirm `bv1.1` and `v1.1-freeze` remain reproducible anchors.

### Milestone 2: Extensibility Refactor

- Introduce data transform pipeline.
- Extract shared FWI engine utilities.
- Keep current public workflows working.

### Milestone 3: Cleaner User Workflow

- Add configuration-driven quickstart examples.
- Add benchmark scripts.
- Improve developer documentation for adding new misfits, regularization methods,
  propagators, and model parameterizations.

### Milestone 4: Performance and Advanced Extensions

Only after the backend layer, tests, and benchmarks are stable:

- checkpoint strategy improvements
- mixed precision experiments on CUDA GPU and NPU when supported
- source encoding improvements
- multi-GPU support
- optional `torch.compile` or JIT experiments

## Acceptance Criteria

`bv1.2` can be considered stable when:

1. Existing core acoustic and elastic ADFWI examples still have a compatibility path.
2. New tests pass on a small CPU environment and on the target NPU system for the NPU smoke-test subset.
3. Adding a new misfit or regularization method does not require editing the main
   inversion loop.
4. Data preprocessing can be configured or composed outside the main FWI class.
5. At least one new quickstart example demonstrates the cleaner `bv1.2` workflow.
6. Benchmark scripts can compare `bv1.1` and `bv1.2` behavior.
7. Device-specific logic is centralized in the backend layer instead of being duplicated across model, propagator, kernel, misfit, regularization, and FWI classes.
8. ADFWI users can configure the default device once, and core ADFWI objects can inherit it without manually passing the same device argument everywhere.

## Notes

The current recommendation is to treat `bv1.2` as an extensibility and
maintainability release. New scientific methods can be added, but they should be
introduced through stable extension points rather than by directly expanding the
main inversion classes. DR-FWI and DIP-related extensions are intentionally deferred
until the ADFWI physical forward/backward path is stable on CPU, CUDA GPU, and NPU.

## bv1.2 Backend API Update

Implemented a user-facing backend entrypoint on top of the lower-level `ADFWI.backends` layer:

```python
import ADFWI

ADFWI.set_backend("npu:0", dtype="float32")
print(ADFWI.backend_diagnostics())
```

The backend API now accepts notebook-friendly string options such as `dtype="float64"` and `prefer="npu,cpu"`, while preserving the existing lower-level `configure_backend`, `get_backend`, and `use_backend` functions. Usage details are recorded in `docs/backend-usage.md`.

## Elastic Mini Inversion Smoke Update

Added `scripts/smoke/elastic_mini_inversion_smoke.py` to cover one-step `ElasticFWI` pressure-component inversion on CPU and NPU. This extends bv1.2 backend validation from elastic forward/backward smoke to the full elastic forward/loss/backward/optimizer/model-update path.

## Data Transform Pipeline Update

Added a phase-1 data transform pipeline plan and standalone pure torch transforms under `ADFWI/fwi/transforms`. This first step covers `TraceNormalize`, `ReceiverMask`, and `DataMask` with CPU/NPU tests, without changing existing `AcousticFWI` or `ElasticFWI` behavior yet. See `docs/version-plans/bv1.2-data-transform-plan.md`.

## AcousticFWI Data Transform Optional Integration

Added an optional `data_transform_pipeline` argument to `AcousticFWI`. The default user behavior is preserved, while `waveform_normalize=True` now builds an internal `DataTransformPipeline([TraceNormalize()])` and bypasses the old `_normalize()` branch in normal FWI execution. Custom pipelines are applied inside `calculate_loss`, enabling gradual migration from embedded waveform processing logic to reusable transforms.
