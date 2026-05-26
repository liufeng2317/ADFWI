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

## Current Direction

- Keep public FWI usage simple while routing internal data preparation through structured helpers.
- Continue moving waveform preprocessing into transform pipelines.
- Preserve FWI numerical behavior first; introduce pure torch/NPU-native
  alternatives only when their numerical differences are explicitly accepted.
- Use smoke tests and small focused unit tests before replacing legacy branches.
