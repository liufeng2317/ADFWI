# bv1.2 Archive Index

This archive groups the detailed `bv1.2` records by theme. The original
numbered files remain the authoritative audit trail and should not be moved or
renumbered.

## Read First

- [Optimization Chain](./optimization-chain.md): top-down architecture,
  ownership map, validation commands, and stabilization boundaries.
- [135 - bv1.2 Closeout Summary](./135-bv12-closeout-summary.md): final branch
  status and stop criteria.
- [00 - Development Plan](./00-development-plan.md): original goals and
  milestones.

## Backend And Public Device Setup

- `01`, `35`-`43`
- Main topic: CPU/CUDA/NPU backend unification, public `ADFWI.set_backend(...)`,
  diagnostics, smoke suites, and minimal examples.

Representative records:

- [01 - Device Backend Design](./01-device-backend-design.md)
- [35 - User-Facing Backend Interface](./35-user-facing-backend-interface.md)
- [37 - Backend Smoke Suite Runner](./37-backend-smoke-suite-runner.md)
- [40 - Backend Example Suite](./40-backend-example-suite.md)
- [43 - Backend Case Check Suite](./43-backend-case-check-suite.md)

## FWI Transform And Pre-Loss Data Handling

- `03`-`07`, `18`-`21`, `28`, `64`-`66`, `95`-`96`, `122`
- Main topic: waveform normalization, receiver selection, masks, mutes,
  low-pass filtering, transform ownership, and removal of thin transform shims.

Representative records:

- [03 - Data Transform Pipeline](./03-data-transform-pipeline.md)
- [06 - Receiver Selection Migration](./06-receiver-selection-migration.md)
- [21 - TraceNormalize Shared Formula](./21-trace-normalize-shared-formula.md)
- [65 - Low-Pass User Guidance](./65-lowpass-user-guidance.md)
- [122 - Transform Responsibility Notes](./122-transform-responsibility-notes.md)

## FWI Iteration And Loss Construction

- `08`-`24`, `26`-`27`, `51`-`63`, `97`-`98`, `116`-`119`
- Main topic: batch scheduling, loss-input records, transform context,
  misfit dispatch, weighted component losses, one-batch steps, epoch updates,
  and removal of the misleading `ADFWI.fwi.data` layer.

Representative records:

- [10 - FWI Iteration Structure](./10-fwi-iteration-structure.md)
- [17 - FWI Loss Pair Builder](./17-fwi-loss-pair-builder.md)
- [59 - FWI Loss Evaluation Helper](./59-fwi-loss-evaluation-helper.md)
- [117 - Merge Data Helpers Into Iteration Loss](./117-merge-data-helpers-into-iteration-loss.md)
- [118 - Iteration Loss Readability Pass](./118-iteration-loss-readability-pass.md)
- [119 - Elastic Loss Components Owner](./119-elastic-loss-components-owner.md)

## Runtime And Driver Ownership

- `29`, `31`-`32`, `48`-`50`, `99`-`100`, `120`-`121`, `134`
- Main topic: shared driver mechanics, backend guards, forward records,
  wavefield accumulation, gradient processor dispatch, regularization
  summation, cache bookkeeping, and final FWI package readability.

Representative records:

- [29 - FWI Runtime Backend Helpers](./29-fwi-runtime-backend-helpers.md)
- [48 - FWI Runtime Cache Helper](./48-fwi-runtime-cache-helper.md)
- [50 - FWI Runtime Gradient Dispatch Helper](./50-fwi-runtime-gradient-dispatch.md)
- [121 - Runtime Responsibility Contract](./121-runtime-responsibility-contract.md)
- [134 - FWI Package Readability Closeout](./134-fwi-package-readability-closeout.md)

## Compatibility Cleanup And Import Surface

- `69`, `93`-`106`
- Main topic: removing thin compatibility shims after `bv1.1` was retained as
  the legacy branch, enforcing canonical owner imports, and documenting public
  surfaces.

Representative records:

- [93 - Compatibility Cleanup Audit](./93-compatibility-cleanup-audit.md)
- [94 - Remove Thin Compatibility Shims](./94-remove-thin-compat-shims.md)
- [104 - Example Import Surface Audit](./104-example-import-surface-audit.md)
- [105 - Import Surface Policy Test](./105-import-surface-policy-test.md)
- [106 - Broaden Import Surface Policy](./106-import-surface-policy-broaden.md)

## Gradient Processor And Performance Gates

- `67`-`76`, `88`-`92`, `107`-`115`
- Main topic: opt-in `TorchGradProcessor`, legacy-vs-torch parity,
  NPU tolerance profile, Marmousi2 stress gates, fixed benchmark presets,
  output comparison, and profiling.

Representative records:

- [68 - Torch Gradient Processor](./68-torch-gradient-processor.md)
- [74 - Acoustic Benchmark Gradient Processors](./74-acoustic-benchmark-gradient-processors.md)
- [91 - Marmousi2 Preset Python Profiling Option](./91-marmousi2-preset-python-profile.md)
- [110 - Gradient Processor NPU Tolerance Profile](./110-gradient-processor-npu-tolerance-profile.md)
- [115 - Torch Gradient Status And Next Direction](./115-torch-gradient-status-and-next-direction.md)

## Marmousi2 Full-Case And Validation Workflows

- `42`-`47`, `75`-`92`, `123`-`133`
- Main topic: realistic Marmousi2 checks, saved output artifacts, preset
  runners, comparison tools, and the final validation example split into
  forward/inversion notebooks and scripts.

Representative records:

- [80 - Full-Case Test Suite](./80-full-case-test-suite.md)
- [89 - Marmousi2 Full-Case Preset Runner](./89-marmousi2-full-case-presets.md)
- [123 - Marmousi2 Validation Example](./123-marmousi2-validation-example.md)
- [131 - Marmousi2 Validation Script Modules](./131-marmousi2-validation-script-modules.md)
- [132 - Marmousi2 Forward Script Comparison](./132-marmousi2-forward-script-comparison.md)
- [133 - Marmousi2 Inversion Script Comparison](./133-marmousi2-inversion-script-comparison.md)

## Documentation And Release Stabilization

- `22`, `33`-`34`, `41`, `72`, `87`, `115`, `121`-`122`, `134`-`135`
- Main topic: module map, optimization chain, responsibility contracts,
  stopping broad optimization, and release/stabilization handoff.

Representative records:

- [33 - FWI Module Map](./33-fwi-module-map.md)
- [72 - Optimization Chain Documentation](./72-optimization-chain-doc.md)
- [87 - Marmousi2 Benchmark Direction Adjustment](./87-marmousi2-benchmark-direction-adjustment.md)
- [135 - bv1.2 Closeout Summary](./135-bv12-closeout-summary.md)

## Deprecated Direction

Do not use the archive as a queue of unfinished tasks. Many older records
describe intermediate designs that were later superseded. Current contracts are
defined by:

- `optimization-chain.md`;
- `135-bv12-closeout-summary.md`;
- the latest code in `ADFWI/fwi`;
- the validation examples under `examples/validation/marmousi2_acoustic_bv12/`.
