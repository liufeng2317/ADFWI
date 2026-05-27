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

## Current Direction

- Keep public FWI usage simple while routing internal data preparation through structured helpers.
- Continue moving waveform preprocessing into transform pipelines.
- Preserve FWI numerical behavior first; introduce pure torch/NPU-native
  alternatives only when their numerical differences are explicitly accepted.
- Use smoke tests and benchmark comparisons before replacing legacy branches.
- Use NPU for reduced real-case multi-iteration gates; keep CPU real-case checks
  short because the 300-sample path is slow.
