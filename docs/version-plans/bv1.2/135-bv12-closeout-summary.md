# 135 - bv1.2 Closeout Summary

## Status

`bv1.2` has reached a framework-stabilization checkpoint. The branch should now
be treated as a release/stabilization branch rather than an open-ended
optimization branch.

The main framework work is complete:

- backend/device/dtype selection is centralized through `ADFWI.backends` and
  the top-level `ADFWI.set_backend(...)` API;
- FWI waveform preprocessing is owned by `ADFWI.fwi.transforms`;
- FWI iteration mechanics are owned by `ADFWI.fwi.iteration`;
- shared execution mechanics are owned by `ADFWI.fwi.runtime`;
- legacy multiscale low-pass behavior is explicit under `ADFWI.fwi.multiscale`;
- acoustic and elastic FWI drivers keep ownership of physical fields,
  inversion components, parameter order, and user-facing configuration;
- the misleading `ADFWI.fwi.data` helper layer has been removed and its useful
  loss-preparation helpers now live under `ADFWI.fwi.iteration.loss`;
- Marmousi2 validation examples now provide separate forward-modeling and
  inversion workflows for both notebooks and scripts.

## What Changed By Layer

| Layer | Closeout State |
| --- | --- |
| Backend | User-facing backend setup is stable. Prefer `ADFWI.set_backend(...)` in new examples. |
| FWI drivers | `AcousticFWI` and `ElasticFWI` remain the public workflow entry points. Driver internals now delegate repeated mechanics to `iteration` and `runtime`. |
| Iteration | Batch scheduling, loss construction, one-batch steps, and epoch updates are separated into readable owner modules. |
| Runtime | Shared driver mechanics are extracted but not over-split. Runtime is not a standalone inversion framework. |
| Transforms | Pre-loss waveform transforms are centralized and structurally stable. Do not reshuffle without a specific bug. |
| Multiscale | Legacy low-pass remains explicit for numerical compatibility. |
| Misfit/regularization | Public exports and abstract contracts are clearer; formulas are unchanged. |
| Validation | Marmousi2 validation scripts and notebooks match, and saved notebook outputs are recorded. |

## Validation State

The branch has accumulated three validation levels:

1. Lightweight unit tests for backend, transforms, iteration, runtime, import
   surface, gradient processing, and output comparison tools.
2. Smoke scripts for public backend/FWI workflows.
3. Opt-in Marmousi2 full-case gates for realistic forward-plus-inversion
   checks.

Important validation conclusions:

- `TorchGradProcessor` is NPU-validated as an opt-in path, not the default.
- Legacy `GradProcessor` remains the default numerical path.
- Marmousi2 forward and inversion validation scripts match the manually
  verified notebooks.
- Further real-case testing should only be required when numerical FWI paths
  change.

## Archive Policy

The numbered records from `00` to `135` are retained as the detailed audit trail.
Do not rewrite or renumber them. For future reading:

- use `optimization-chain.md` for the top-down architecture and current
  contracts;
- use `archive-index.md` for grouped access to the detailed records;
- use this closeout summary to understand the final state of the branch.

## Stop Criteria

The current optimization stream should stop here unless a new task has a
specific boundary. Avoid more broad cleanup. New work should be one of:

- release notes or merge preparation;
- a targeted example migration;
- a targeted elastic validation case;
- a measured performance bottleneck;
- a concrete failing test or bug.

## Recommended Next Tasks

Recommended follow-up order:

1. Write bv1.2 release notes from this closeout summary and
   `optimization-chain.md`.
2. Decide whether to migrate one main acoustic example to the new explicit
   backend/import style.
3. Optionally add one elastic validation case, mirroring the Marmousi2 acoustic
   validation structure.
4. Run the lightweight stabilization test set before merge/tag.

Do not continue adding optimization records unless the change modifies behavior,
public workflow, validation coverage, or release documentation.
