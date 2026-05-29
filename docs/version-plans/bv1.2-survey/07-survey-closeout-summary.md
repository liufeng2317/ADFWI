# 07 - Survey Closeout Summary

## Status

The `ADFWI/survey` bv1.2 cleanup is closed.

This stage clarified ownership and contracts for source/receiver geometry,
survey masks, recorded waveform storage, save/load, parse helpers, and plotting
helpers without changing propagator kernels, FWI iteration logic, or example
calling style.

## Final Ownership

| Module | Owns | Does not own |
| --- | --- | --- |
| `source.py` | source time axis, grid-index source locations, wavelets, moment tensors, source type metadata | receiver geometry, waveform data, propagator logic |
| `receiver.py` | receiver time axis, grid-index receiver locations, receiver component type metadata | source wavelets, waveform data, propagator logic |
| `survey.py` | acquisition state combining `Source` and `Receiver`, receiver masks, geometry plotting helpers | waveform arrays, model data, FWI loss, propagator execution |
| `data.py` | survey metadata snapshot, recorded waveform dictionary, `.npz` save/load, acoustic/elastic parse helpers, waveform plotting helpers | source/receiver mutation, propagator kernels, FWI loss construction |

## Completed Optimization Chain

1. Responsibility readability:
   - added concise module/class docstrings;
   - removed stale comments/import noise;
   - established `Survey` as acquisition state and `SeismicData` as waveform snapshot.
2. `SeismicData` contracts:
   - `record_data()` now stores an internal numpy copy instead of mutating caller input;
   - save/load round trip is covered;
   - acoustic and elastic parse outputs are covered by exact deterministic tests;
   - zero-trace normalization behavior is covered.
3. Receiver mask contract:
   - masks are explicitly 2D `[source, receiver]`;
   - shape must match `(source.num, receiver.num)`;
   - `receiver_masks_obs` is preserved as a downstream observed-data mask flag.
4. Source/Receiver input contract:
   - normal source and receiver bulk-add locations are 1D array-like inputs;
   - source wavelets must match `nt`;
   - encoded source wavelets preserve leading encoded axes while requiring last dimension `nt`;
   - invalid inputs now raise clear `ValueError`.
5. Survey/SeismicData boundary:
   - parse helpers now check recorded state and required components before indexing;
   - legal acoustic/elastic numerical behavior is unchanged.
6. Plot helper contract:
   - `plot_single_shot()` now passes 1D active receiver coordinates to `plot_survey()`;
   - plotting helper change does not affect propagator/FWI paths.

## Validation Used

The survey-stage changes were validated with focused tests and integration tests:

| Validation | Purpose |
| --- | --- |
| `tests/test_survey_contracts.py` | source/receiver/survey/data shape, save/load, parse, mask, and plotting helper contracts |
| `tests/test_backend_integration.py` | backend and receiver-mask integration paths |
| `tests/test_receiver_selection.py` | receiver mask selection behavior during mask contract work |
| `py_compile ADFWI/survey/*.py tests/test_survey_contracts.py` | import/compile safety |
| Marmousi2 validation `check` stage | real-case source/receiver/survey construction smoke after caller-facing contract changes |
| `git diff --check` | whitespace and patch hygiene |

## Remaining Risks

These are known design risks, not active cleanup tasks.

| Risk | Current decision | When to revisit |
| --- | --- | --- |
| `Source.get_type(unique=True)` and `Receiver.get_type(unique=True)` use `set`, so unique order is not deterministic | left unchanged for compatibility because normal callers use non-unique type arrays | revisit only if a workflow depends on stable unique ordering |
| `.npz` save/load stores waveform dict as pickle object | left unchanged to avoid breaking existing data artifacts | revisit only with an explicit storage compatibility plan |
| `SeismicData` still includes plot helpers | kept for example/API compatibility | revisit only if a broader visualization API is planned |
| waveform orientation in plotting helpers uses existing transpose conventions | left unchanged because visual orientation changes require screenshot or reference-output validation | revisit only with a dedicated plotting validation case |
| broad `__repr__` exception handling in `Source`/`Receiver` remains | left unchanged because it preserves empty-object display behavior | revisit only with focused empty/non-empty repr tests |
| encoded source semantics remain specialized | documented and lightly tested but not redesigned | revisit only if source-encoding workflows become an active validation target |

## Stop Rule

Do not continue micro-optimizing `ADFWI/survey` after this closeout unless one of
the remaining risks becomes a concrete bug or blocks a validation workflow.

Future survey work should start from a new bounded task with:

```text
Goal:
Scope:
Validation:
Stop:
```
