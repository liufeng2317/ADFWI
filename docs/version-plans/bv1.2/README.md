# bv1.2 Planning Index

`bv1.2` is the active development branch for framework cleanup, backend/device
unification, data transforms, tests, and benchmark preparation. Read the files in
numbered order unless you are looking for a specific topic.

## Documents

| Order | Document | Purpose |
| --- | --- | --- |
| 00 | [Development Plan](./00-development-plan.md) | Branch goals, non-goals, milestones, and current status. |
| 01 | [Device Backend Design](./01-device-backend-design.md) | Unified CPU/CUDA/NPU backend and user-facing device configuration. |
| 02 | [Misfit Backend Audit](./02-misfit-backend-audit.md) | Misfit CPU/NPU compatibility audit and refactor notes. |
| 03 | [Data Transform Pipeline](./03-data-transform-pipeline.md) | Transform pipeline design and migration status for masks, normalization, and filtering. |
| 04 | [Low-Pass Filter Comparison](./04-lowpass-filter-comparison.md) | Legacy vs torch low-pass validation, inversion smoke comparison, and precision decision. |
| 05 | [Mute Transform Migration](./05-mute-transform-migration.md) | Offset and first-arrival mute migration into the shared transform pipeline. |
| 06 | [Receiver Selection Migration](./06-receiver-selection-migration.md) | Centralized receiver mask and trace-missing selection before loss calculation. |
| 07 | [Transform Module Organization](./07-transform-module-organization.md) | Module split for transform implementations while preserving public imports. |
| 08 | [FWI Data Contract](./08-fwi-data-contract.md) | Pre-loss synthetic/observed data contract and shared preparation helper. |
| 09 | [FWI User-Facing Interface](./09-fwi-user-facing-interface.md) | Recommended public usage for transforms, elastic components, and component weights. |
| 10 | [FWI Iteration Structure](./10-fwi-iteration-structure.md) | Shared batch range helper and staged cleanup plan for acoustic/elastic FWI iteration flow. |
| 11 | [FWI Cache Organization](./11-fwi-cache-organization.md) | Acoustic cache helper cleanup and validation plan for result bookkeeping. |
| 12 | [FWI Regularization Organization](./12-fwi-regularization-organization.md) | Acoustic/elastic model regularization helper extraction and validation plan. |
| 13 | [FWI Loss Accumulator](./13-fwi-loss-accumulator.md) | Shared batch loss tensor/scalar helper and validation plan. |
| 14 | [FWI Batch Progress](./14-fwi-batch-progress.md) | Shared single-batch progress description helper and validation plan. |
| 15 | [FWI Transform Pipeline Builder](./15-fwi-transform-pipeline-builder.md) | Shared default transform pipeline builder and validation plan. |
| 16 | [FWI Transform Context Builder](./16-fwi-transform-context-builder.md) | Shared transform context builder and validation plan. |
| 17 | [FWI Loss Pair Builder](./17-fwi-loss-pair-builder.md) | Shared pre-loss pair builder and validation plan. |
| 18 | [Calculate Loss Receiver Selection](./18-calculate-loss-receiver-selection.md) | Direct calculate_loss receiver selection and validation plan. |
| 19 | [FWI Waveform Normalization Helper](./19-fwi-waveform-normalization.md) | Shared legacy waveform normalization helper and validation plan. |
| 20 | [FWI Misfit Evaluation Helper](./20-fwi-misfit-evaluation.md) | Shared misfit dispatch helper and validation plan. |
| 21 | [TraceNormalize Shared Formula](./21-trace-normalize-shared-formula.md) | Shared normalization formula for TraceNormalize and FWI fallback paths. |
| 22 | [Documentation Consistency Pass](./22-doc-consistency-pass.md) | Align earlier bv1.2 records with current data-path behavior. |
| 23 | [Elastic Component Loss Inputs](./23-elastic-component-loss-inputs.md) | Shared elastic component input selection before loss evaluation. |
| 24 | [Weighted Loss Sum Helper](./24-weighted-loss-sum.md) | Shared weighted loss summation helper for component losses. |
| 25 | [Elastic Backend Guard](./25-elastic-backend-guard.md) | ElasticFWI device consistency and regularization backend alignment. |
| 26 | [FWI Data Package Organization](./26-fwi-data-package-organization.md) | Split `ADFWI.fwi.data` into responsibility-focused package modules while preserving public imports. |
| 27 | [FWI Iteration Package Organization](./27-fwi-iteration-package-organization.md) | Split `ADFWI.fwi.iteration` into batch range, loss, and progress modules while preserving public imports. |
| 28 | [Waveform Normalization Ownership](./28-waveform-normalization-ownership.md) | Move canonical waveform normalization formula next to `TraceNormalize` while preserving compatibility imports. |
| 29 | [FWI Runtime Backend Helpers](./29-fwi-runtime-backend-helpers.md) | Share FWI constructor backend guards and regularization backend alignment helpers. |
| 30 | [FWI Multiscale Package Organization](./30-fwi-multiscale-package-organization.md) | Move legacy multiscale low-pass implementation into a package while preserving `multiScaleProcessing` imports. |
| 31 | [FWI Runtime Regularization Helper](./31-fwi-runtime-regularization-helper.md) | Share the single-parameter regularization loss rule used by AcousticFWI and ElasticFWI. |
| 32 | [FWI Runtime Gradient Helper](./32-fwi-runtime-gradient-helper.md) | Share AcousticFWI/ElasticFWI gradient processor dispatch while preserving legacy gradient numerics. |
| 33 | [FWI Module Map for Geophysical Users](./33-fwi-module-map.md) | Map the bv1.2 FWI package structure to the physical inversion workflow for future refactors. |
| 34 | [Core Comment and Contract Pass](./34-core-comment-contract-pass.md) | Clarify core helper docstrings, variable meanings, and type contracts without changing numerical behavior. |
| 35 | [User-Facing Backend Interface](./35-user-facing-backend-interface.md) | Stabilize documented backend entry points and tests for one-line CPU/NPU/CUDA setup. |
| 36 | [Backend Public API Smoke](./36-backend-public-api-smoke.md) | Add a lightweight command-line smoke for top-level backend configuration and diagnostics. |
| 37 | [Backend Smoke Suite Runner](./37-backend-smoke-suite-runner.md) | Add one layered JSON runner for public API, misfit, forward, and mini-inversion backend smoke checks. |
| 38 | [Minimal Acoustic Backend Example](./38-minimal-acoustic-backend-example.md) | Add a no-output script example showing the recommended backend setup through one tiny AcousticFWI iteration. |
| 39 | [Minimal Elastic Backend Example](./39-minimal-elastic-backend-example.md) | Add a no-output script example showing the recommended backend setup through one tiny pressure-component ElasticFWI iteration. |
| 40 | [Backend Example Suite](./40-backend-example-suite.md) | Add the user-facing acoustic/elastic minimal examples to the layered backend smoke runner. |
| 41 | [Script Example Guide](./41-script-example-guide.md) | Add a researcher-facing guide for running and adapting script-style backend examples. |
| 42 | [Marmousi2 Script Backend Check](./42-marmousi2-script-backend-check.md) | Add a read-only script entry point for checking the existing acoustic Marmousi2 case on CPU/NPU. |
| 43 | [Backend Case Check Suite](./43-backend-case-check-suite.md) | Add read-only real-case checks to the layered backend smoke runner. |
| 44 | [Marmousi2 Forward Case Check](./44-marmousi2-forward-case-check.md) | Add optional single-shot forward validation and CPU/NPU norm comparison for the Marmousi2 acoustic case. |
| 45 | [Marmousi2 Reduced Inversion Smoke](./45-marmousi2-reduced-inversion-smoke.md) | Add a no-output Marmousi2 subset inversion script to validate backward and gradient paths. |
| 46 | [Legacy L2 Numerical Stability](./46-legacy-l2-numerical-stability.md) | Record the zero-residual NaN gradient risk in the historical L2-norm misfit and design a stable squared-L2 migration path. |
| 47 | [Backend Case Inversion Suite](./47-backend-case-inversion-suite.md) | Add reduced Marmousi2 real-case inversion checks to the layered backend smoke runner with CPU/NPU metric comparison. |
| 48 | [FWI Runtime Cache Helper](./48-fwi-runtime-cache-helper.md) | Share AcousticFWI/ElasticFWI result-cache bookkeeping while keeping physical parameter choices in each driver. |
| 49 | [FWI Runtime Wavefield Helper](./49-fwi-runtime-wavefield-helper.md) | Share forward-wavefield accumulation for gradient processors while preserving acoustic/elastic component semantics. |
| 50 | [FWI Runtime Gradient Dispatch Helper](./50-fwi-runtime-gradient-dispatch.md) | Share trainable-parameter gradient dispatch while keeping physical parameter lists in AcousticFWI and ElasticFWI. |
| 51 | [FWI Iteration Batch Step Helper](./51-fwi-iteration-batch-step.md) | Share batch loss/backward/progress bookkeeping while keeping data-loss construction in FWI drivers. |
| 52 | [FWI Iteration Epoch Update Helper](./52-fwi-iteration-epoch-update.md) | Share optimizer/scheduler/model-constraint epoch-tail ordering while keeping closure construction in FWI drivers. |
| 53 | [FWI Iteration Epoch Finalization Helper](./53-fwi-iteration-epoch-finalization.md) | Share epoch cache/progress finalization while keeping model-specific result saving in FWI drivers. |
| 54 | [FWI Batch Range Cleanup](./54-fwi-batch-range-cleanup.md) | Remove stale batch-bound locals from acoustic/elastic iteration loops while preserving shot selection and progress metadata. |
| 55 | [FWI Waveform Alias Cleanup](./55-fwi-waveform-alias-cleanup.md) | Remove unused waveform aliases from acoustic/elastic batch loops while preserving record-level loss construction. |
| 56 | [FWI Wavefield Input Helper](./56-fwi-wavefield-input-helper.md) | Share small record-to-wavefield selection helpers while preserving acoustic/elastic loss semantics. |
| 57 | [FWI Forward Batch Record](./57-fwi-forward-batch-record.md) | Introduce a per-batch forward record so shot selection and propagator output move together through FWI loops. |
| 58 | [FWI Loss Input Record](./58-fwi-loss-input-record.md) | Move raw synthetic/observed loss-input pairing into the data contract layer while preserving transform and misfit behavior. |
| 59 | [FWI Loss Evaluation Helper](./59-fwi-loss-evaluation-helper.md) | Share prepared loss-input evaluation and weighted component accumulation while keeping physical loss-input choices in the drivers. |
| 60 | [Acoustic Batch Loss Step Helper](./60-acoustic-batch-loss-step.md) | Share the acoustic per-batch forward/loss/backward body between closure and non-closure optimizer paths. |
| 61 | [Elastic Batch Loss Step Helper](./61-elastic-batch-loss-step.md) | Share the elastic per-batch forward/component-loss/backward body while preserving component and wavefield semantics. |
| 62 | [FWI Parameter Spec Helpers](./62-fwi-parameter-spec-helpers.md) | Centralize acoustic/elastic parameter names and gradient processor index specs while keeping model choices explicit. |
| 63 | [FWI Model Regularization Sum Helper](./63-fwi-model-regularization-sum.md) | Share ordered model-parameter regularization summation for acoustic and elastic drivers. |
| 64 | [Legacy Low-Pass Ownership Comments](./64-legacy-lowpass-ownership-comments.md) | Clarify multiscale low-pass implementation ownership and compatibility imports without changing numerical behavior. |
| 65 | [Low-Pass User Guidance](./65-lowpass-user-guidance.md) | Add user-facing guidance for choosing legacy-compatible versus pure torch low-pass filtering. |
| 66 | [Receiver Selection Torch Indexing](./66-receiver-selection-torch-indexing.md) | Keep trace-missing receiver selection on the active torch device while preserving legacy receiver order. |
| 67 | [Acoustic Benchmark Scaffold](./67-acoustic-benchmark-scaffold.md) | Add a reproducible tiny acoustic benchmark JSON runner before performance-oriented changes. |
| 68 | [Torch Gradient Processor](./68-torch-gradient-processor.md) | Add an opt-in torch-native gradient processor while preserving legacy GradProcessor behavior. |
| 69 | [Compatibility Shim Import Cleanup](./69-compat-shim-import-cleanup.md) | Route internal imports to canonical owners while keeping thin compatibility shims for old scripts. |
| 70 | [Acoustic Torch Gradient Smoke](./70-acoustic-torch-gradient-smoke.md) | Add an opt-in TorchGradProcessor path to the acoustic mini-inversion smoke with legacy drift checks. |
| 71 | [Example Torch Gradient Option](./71-example-torch-gradient-option.md) | Expose opt-in TorchGradProcessor in minimal examples and the examples smoke suite with CPU/NPU drift checks. |
| 72 | [Optimization Chain Documentation](./72-optimization-chain-doc.md) | Add a top-level optimization chain diagram and summary under `docs/`. |
| 73 | [Example Gradient Processor Comparison](./73-example-gradient-processor-comparison.md) | Let the examples smoke suite run legacy and torch gradient processors together and report drift. |
| 74 | [Acoustic Benchmark Gradient Processors](./74-acoustic-benchmark-gradient-processors.md) | Extend the acoustic benchmark to time and compare legacy versus torch gradient processors. |
| 75 | [Marmousi2 Iteration Smoke Control](./75-marmousi2-iteration-smoke-control.md) | Add configurable reduced real-case iteration counts and validate a multi-iteration Marmousi2 smoke. |
| 76 | [Marmousi2 NPU 10-Iteration Gate](./76-marmousi2-npu-10-iteration-gate.md) | Record the first reduced Marmousi2 10-iteration real-case gate on NPU. |
| 77 | [Marmousi2 Loss Sensitivity Diagnostics](./77-marmousi2-loss-sensitivity-diagnostics.md) | Explain unchanged reduced-case loss and add explicit loss-delta reporting. |
| 78 | [Marmousi2 Notebook Settings Check](./78-marmousi2-notebook-settings-check.md) | Compare reduced real-case settings against the original Marmousi2 notebook configuration. |
| 79 | [Marmousi2 Synthetic Observation Check](./79-marmousi2-synthetic-observation-check.md) | Generate true-model observations with current code and validate notebook-like inversion settings. |
| 80 | [Full-Case Test Suite](./80-full-case-test-suite.md) | Add opt-in full-case tests for forward-plus-inversion real-case validation. |
| 81 | [Full-Case Output Artifacts](./81-full-case-output-artifacts.md) | Save JSON, CSV, and PNG artifacts from opt-in Marmousi2 full-case runs for visual inspection. |
| 82 | [Marmousi2 NPU Checkpoint-1 10-Iteration Test](./82-marmousi2-npu-checkpoint1-10iter.md) | Compare full-case NPU runtime and loss behavior with `checkpoint_segments=1`. |
| 83 | [Marmousi2 NPU 3-Shot Checkpoint-1 Test](./83-marmousi2-npu-shot3-checkpoint1.md) | Validate a 3-shot full-length NPU gate with `checkpoint_segments=1` and compare per-iteration efficiency. |
| 84 | [Marmousi2 NPU 3-Shot 10-Iteration Gate](./84-marmousi2-npu-shot3-10iter.md) | Confirm stable 3-shot NPU efficiency and monotonic loss over 10 full-length iterations. |
| 85 | [Marmousi2 NPU 3-Shot Checkpoint-10 Comparison](./85-marmousi2-npu-shot3-checkpoint10.md) | Compare 3-shot full-case checkpoint segmentation and select the faster current NPU benchmark baseline. |
| 86 | [Marmousi2 NPU 5-Shot Checkpoint-10 Gate](./86-marmousi2-npu-shot5-checkpoint10.md) | Probe the next NPU throughput step with a 5-shot full-length checkpoint-10 benchmark. |
| 87 | [Marmousi2 Benchmark Direction Adjustment](./87-marmousi2-benchmark-direction-adjustment.md) | Stop the shot-count sweep and use existing 3-shot/5-shot gates as fixed baselines for later optimization. |
| 88 | [Full-Case Output Compare Tool](./88-full-case-output-compare-tool.md) | Add a saved-output comparison CLI for fixed Marmousi2 full-case baselines. |
| 89 | [Marmousi2 Full-Case Preset Runner](./89-marmousi2-full-case-presets.md) | Add reusable preset commands for the fixed 3-shot and 5-shot Marmousi2 full-case baselines. |
| 90 | [Marmousi2 Preset Post-Run Compare](./90-marmousi2-preset-postrun-compare.md) | Let preset full-case runs compare against saved baselines automatically after completion. |
| 91 | [Marmousi2 Preset Python Profiling Option](./91-marmousi2-preset-python-profile.md) | Add optional cProfile wrapping to fixed full-case presets before deeper performance work. |
| 92 | [Marmousi2 Shot3 Python Profile Run](./92-marmousi2-shot3-python-profile-run.md) | Run the fixed shot3 baseline under cProfile and identify autograd/propagator paths as the next profiling target. |
| 93 | [Compatibility Cleanup Audit](./93-compatibility-cleanup-audit.md) | Audit retained compatibility shims and legacy numerical paths before staged cleanup. |
| 94 | [Remove Thin Compatibility Shims](./94-remove-thin-compat-shims.md) | Remove old normalization and multiScaleProcessing import shims in favor of canonical bv1.2 APIs. |
| 95 | [Remove Transform Waveform Shim](./95-remove-transform-waveform-shim.md) | Remove the unused `ADFWI.fwi.transforms.waveform` re-export module. |
| 96 | [Remove Data Normalize Re-Export](./96-remove-data-normalize-reexport.md) | Remove `ADFWI.fwi.data.normalize_waveform` in favor of transform-layer canonical imports. |
| 97 | [Iteration Owner Imports](./97-iteration-owner-imports.md) | Route active FWI iteration imports to owner modules before narrowing package-level re-exports. |
| 98 | [Remove Iteration Re-Exports](./98-remove-iteration-reexports.md) | Remove broad `ADFWI.fwi.iteration` helper re-exports after active code moved to owner modules. |
| 99 | [Runtime Owner Imports](./99-runtime-owner-imports.md) | Route active FWI runtime imports to backend/cache/forward/gradient/regularization/wavefield owner modules. |
| 100 | [Remove Runtime Re-Exports](./100-remove-runtime-reexports.md) | Remove broad `ADFWI.fwi.runtime` helper re-exports after active code moved to owner modules. |
| 101 | [Data Owner Imports](./101-data-owner-imports.md) | Route active FWI data-contract imports to components/inputs/loss/pipeline/preparation owner modules. |
| 102 | [Data Public Facade](./102-data-public-facade.md) | Keep `ADFWI.fwi.data` as a deliberate public facade while internals use owner modules. |
| 103 | [Data Facade User Documentation](./103-data-facade-user-doc.md) | Document the public `ADFWI.fwi.data` facade and namespace-only internal helper packages. |
| 104 | [Example Import Surface Audit](./104-example-import-surface-audit.md) | Update examples and sphinx API sources that still referenced removed import shims. |
| 105 | [Import Surface Policy Test](./105-import-surface-policy-test.md) | Add a tracked-file regression test for removed shims and namespace-only import surfaces. |
| 106 | [Broaden Import Surface Policy](./106-import-surface-policy-broaden.md) | Catch alternate namespace-only package imports in the import-surface policy test. |
| 107 | [Torch Gradient Parity Convergence](./107-torch-gradient-parity-convergence.md) | Fix legacy land taper compatibility and expand legacy-vs-torch gradient processor parity tests. |
| 108 | [Gradient Processor Timing Gate](./108-gradient-processor-timing-gate.md) | Add a focused JSON benchmark for legacy-vs-torch gradient processor parity and timing. |
| 109 | [Gradient Processor Stage Diagnostics](./109-gradient-processor-stage-diagnostics.md) | Isolate NPU gradient processor drift by smoothing, normalization, and illumination stages. |
| 110 | [Gradient Processor NPU Tolerance Profile](./110-gradient-processor-npu-tolerance-profile.md) | Add an explicit NPU float32 tolerance profile while keeping strict parity as the default. |
| 111 | [Marmousi2 Torch Gradient Opt-In](./111-marmousi2-torch-gradient-opt-in.md) | Wire TorchGradProcessor into fixed Marmousi2 full-case entry points without changing the default. |
| 112 | [Marmousi2 Gradient Stress Gates](./112-marmousi2-gradient-stress-gates.md) | Add real-case gradient stress controls and validate smoothing/illumination torch paths on NPU. |
| 113 | [Marmousi2 Shot3 Smoothing Trajectory](./113-marmousi2-shot3-smoothing-trajectory.md) | Validate a 3-shot, 10-iteration NPU smoothing trajectory for legacy versus torch gradient processors. |
| 114 | [Marmousi2 Shot3 Illumination Trajectory](./114-marmousi2-shot3-illumination-trajectory.md) | Validate a 3-shot, 10-iteration NPU illumination trajectory for legacy versus torch gradient processors. |
| 115 | [Torch Gradient Status And Next Direction](./115-torch-gradient-status-and-next-direction.md) | Mark TorchGradProcessor as NPU-validated opt-in and redirect later work away from more gradient tests. |
| 116 | [FWI Data Context Helper](./116-fwi-data-context-helper.md) | Extract shot-scoped transform-context selection inside FWI data preparation without changing behavior. |
| 117 | [Merge Data Helpers Into Iteration Loss](./117-merge-data-helpers-into-iteration-loss.md) | Remove the misleading `ADFWI.fwi.data` package and move its helpers into iteration-owned batch loss construction. |
| 118 | [Iteration Loss Readability Pass](./118-iteration-loss-readability-pass.md) | Clarify `iteration/loss.py` sections and rename the loss transform pipeline builder away from old data wording. |
| 119 | [Elastic Loss Components Owner](./119-elastic-loss-components-owner.md) | Move supported elastic loss components into `elastic_fwi.py` and pass them into iteration loss helpers. |
| 120 | [Gradient Parameter Ownership](./120-gradient-parameter-ownership.md) | Move acoustic/elastic parameter-order constants out of runtime gradient dispatch and into their FWI drivers. |
| 121 | [Runtime Responsibility Contract](./121-runtime-responsibility-contract.md) | Clarify runtime module responsibilities without changing execution behavior. |
| 122 | [Transform Responsibility Notes](./122-transform-responsibility-notes.md) | Clarify transform module responsibilities without changing behavior. |
| 123 | [Marmousi2 Validation Example](./123-marmousi2-validation-example.md) | Add separate Python/Jupyter validation entry points for staged Marmousi2 checks. |
| 124 | [Marmousi2 Validation Notebook Split](./124-marmousi2-validation-notebook-split.md) | Separate forward-modeling and inversion validation notebooks. |
| 125 | [Marmousi2 Validation Parameter Notebooks](./125-marmousi2-validation-parameter-notebooks.md) | Expose parameters, model, survey, and wavelet definitions in validation notebooks. |
| 126 | [Marmousi2 Validation Local Case Definition](./126-marmousi2-validation-local-case-definition.md) | Keep validation notebooks from importing case definitions from another case/check script. |
| 127 | [Marmousi2 Validation Inline Notebook Definitions](./127-marmousi2-validation-inline-notebook-definitions.md) | Inline lightweight case definitions in validation notebooks for readability. |

## Current Direction

- Keep public FWI usage simple while routing internal data preparation through structured helpers.
- Continue moving waveform preprocessing into transform pipelines.
- Preserve FWI numerical behavior first; introduce pure torch/NPU-native
  alternatives only when their numerical differences are explicitly accepted.
- Use smoke tests and benchmark comparisons before replacing legacy branches.
- Use NPU for reduced real-case multi-iteration gates; keep CPU real-case checks
  short because the 300-sample path is slow.
- Keep heavy forward-plus-inversion real-case tests under `tests/full_cases/`
  and require explicit opt-in.
- Use 3-shot full-length NPU gates as the next realistic benchmark step because
  they run faster per iteration than the 1-shot gate while preserving stable
  synthetic-true loss behavior.
- Treat the 3-shot, 10-iteration, `checkpoint_segments=10` run as the current
  full-case NPU baseline because it is faster than `checkpoint_segments=1` with
  equivalent loss behavior.
- Use 5-shot, 10-iteration, `checkpoint_segments=10` as the current throughput
  stress baseline; it remains stable and only slightly slower per iteration than
  the 3-shot run.
- Do not continue the shot-count sweep for now; use the existing 3-shot and
  5-shot full-case gates to validate concrete code or workflow optimizations.
- Use `scripts/benchmark/compare_full_case_outputs.py` to compare saved
  full-case outputs after future code or workflow changes.
- Use `scripts/benchmark/run_marmousi2_full_case.py` to rerun the fixed 3-shot
  and 5-shot baselines without manually copying long commands.
- Use preset `--compare-to` when validating future optimizations so every run
  can immediately compare against a saved baseline.
- Use preset `--profile` to separate Python-side FWI overhead from propagator
  and NPU runtime before changing core performance paths.
- Python profiling shows the fixed shot3 baseline is dominated by autograd
  backward and checkpointed propagator execution, so next performance work
  should use torch/NPU operator-level profiling.
- Start compatibility cleanup by removing thin import shims; do not remove or
  replace legacy numerical methods without focused tests and full-case
  comparison.
- Because `bv1.1` is retained for legacy compatibility, `bv1.2` now removes thin
  import shims while preserving explicit legacy numerical methods.
- Continue removing pure re-export modules when active code has already moved
  to canonical bv1.2 import paths.
- Keep waveform operations owned by `ADFWI.fwi.transforms`; `iteration.loss`
  should only orchestrate loss-input pairing, transform execution, misfit
  dispatch, and weighted loss construction.
- Route active FWI drivers and tests to owner modules under
  `ADFWI.fwi.iteration` before deciding whether package-level iteration
  re-exports should remain public in `bv1.2`.
- Treat `ADFWI.fwi.iteration` as a namespace package in bv1.2; import concrete
  helpers from `batches`, `loss`, `step`, and `epoch`.
- Route active runtime imports to owner modules before deciding whether
  `ADFWI.fwi.runtime` should remain a broad aggregation surface.
- Treat `ADFWI.fwi.runtime` as a namespace package in bv1.2; import concrete
  helpers from backend/cache/forward/gradient/regularization/wavefield modules.
- Treat the former FWI data-contract helpers as part of `ADFWI.fwi.iteration.loss`.
- Treat `ADFWI.fwi.iteration.batches` as the owner of shot batch scheduling.
- Treat `ADFWI.fwi.iteration.step` as the owner of one-batch
  forward/loss/backward execution.
- Keep examples and generated API sources aligned with canonical bv1.2 import
  paths after removing compatibility shims.
- Use `tests/test_import_surface_policy.py` as the lightweight guard for future
  import-surface cleanup.
- Keep the import-surface policy broad enough to catch equivalent `import ...`
  and `from ADFWI.fwi import ...` forms, not only direct helper imports.
- Shift from broad cleanup to convergence-focused optimization: strengthen
  torch gradient parity and timing gates before changing default FWI paths.
- NPU gradient processor drift is localized to the float32 `conv2d` smoothing
  stage and then amplified by `vmax` normalization; keep `TorchGradProcessor`
  opt-in until the NPU tolerance or a dedicated smoothing alternative is chosen.
- Use `--tolerance-profile npu-float32` only for intentional NPU
  `TorchGradProcessor` comparisons; the benchmark default remains strict.
- Marmousi2 full-case entry points now accept `--gradient-processor torch`,
  but default to legacy; the first full-length one-iteration NPU smoke matches
  legacy exactly under the current mask-only gradient settings.
- Real Marmousi2 one-iteration NPU stress gates for `--grad-smooth 2` and
  `--forw-illumination` show no loss drift and only about `1e-6` relative
  gradient/update drift, so the next decision point is a short multi-iteration
  trajectory comparison rather than more single-step diagnostics.
- The 3-shot, 10-iteration `--grad-smooth 2` NPU trajectory is stable for
  legacy versus torch gradient processors, with final-loss relative drift around
  `3e-6` and final update-norm relative drift around `1e-5`.
- The 3-shot, 10-iteration `--forw-illumination` NPU trajectory is also stable,
  with identical final loss and final update-norm relative drift around `1e-7`.
- Treat `TorchGradProcessor` as NPU-validated opt-in, not as the default path.
  Stop expanding gradient-processor tests unless gradient post-processing code
  changes; move next optimization toward operator profiling, release
  stabilization, or reusable real-case configuration.
- Keep further iteration loss cleanup behavior-preserving; do not change
  receiver selection order, transform order, or loss-input shapes without a
  numerical comparison.
- Keep `ADFWI.fwi.runtime.gradient` focused on generic gradient processor
  dispatch; physical acoustic/elastic parameter ownership belongs to the FWI
  drivers.
- Treat `ADFWI.fwi.runtime` as shared driver mechanics, not as a standalone
  forward/inversion framework; stop reshuffling it unless a concrete duplicate
  path or bug appears.
- Keep `ADFWI.fwi.transforms` structurally stable; its current split is
  reasonable, and future work should prefer examples or usage docs over module
  reshuffling.
- Use `examples/validation/marmousi2_acoustic_bv12/` as the staged Python and
  Jupyter entry point for Marmousi2 instance validation without modifying the
  original example notebooks.
