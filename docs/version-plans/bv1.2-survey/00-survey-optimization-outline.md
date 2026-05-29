# 00 - Survey Optimization Outline

## Goal

Improve the clarity and robustness of `ADFWI/survey` while preserving current
public usage and numerical behavior by default.

The survey layer should clearly answer:

- what belongs to `Source`;
- what belongs to `Receiver`;
- what belongs to `Survey`;
- what belongs to `SeismicData`;
- what shape and order are expected by propagators and FWI;
- how receiver masks are represented and applied;
- how saved observed/synthetic data should be loaded back reproducibly.

## Scope

Allowed files for the first survey stage:

- `ADFWI/survey/source.py`
- `ADFWI/survey/receiver.py`
- `ADFWI/survey/survey.py`
- `ADFWI/survey/data.py`
- `ADFWI/survey/__init__.py`
- focused tests under `tests/` if needed
- records under `docs/version-plans/bv1.2-survey/`

Non-scope:

- propagator kernels and padding;
- FWI loop behavior;
- model parameter logic;
- migration of old examples/notebooks;
- changing `.npz` storage format unless a dedicated compatibility plan exists.

## Current Observations

The survey layer is compact and widely used:

- `Source` owns time sampling, source locations, source type, wavelets, and
  moment tensors.
- `Receiver` owns receiver locations, receiver type, and receiver time sampling.
- `Survey` combines one `Source` and one `Receiver`, and optionally holds
  receiver masks.
- `SeismicData` snapshots survey metadata, stores recorded waveform dictionaries,
  saves/loads `.npz`, parses acoustic/elastic components, and calls waveform
  plotting helpers.

Important current behavior:

- Source and receiver coordinates are grid indices, not physical coordinates.
- Most examples assume all shots share the same receiver array.
- Acoustic data uses keys such as `p`, `u`, `w`.
- Elastic data uses keys such as `txx`, `tzz`, `txz`, `vx`, `vz`.
- Saved `.npz` files currently store the waveform dictionary as an object array
  and therefore require `allow_pickle=True` on load.
- Receiver masks are kept in `Survey`, while `SeismicData` is responsible for
  storing and plotting recorded arrays.

## Audit Findings

Potential cleanup areas, ordered by risk:

1. Low-risk readability:
   - stale file headers can become concise module docstrings;
   - duplicate imports in `source.py`;
   - docstrings say "source" in some receiver methods;
   - typo in receiver-mask error message;
   - broad `except` in `__repr__` methods hides unrelated bugs.
2. Contract clarity:
   - source wavelet shape differs across single/multiple/encoded source paths;
   - `get_loc()` returns `(n, 2)` for normal sources but supports encoded
     higher-rank arrays;
   - `get_type(unique=True)` uses `set`, so order is not deterministic;
   - `SeismicData.record_data()` mutates the input dictionary while converting
     tensors to numpy arrays.
3. Behavior-sensitive areas:
   - `.npz` object-dict storage and `allow_pickle=True`;
   - receiver mask semantics and whether observed data should be masked;
   - acoustic/elastic component naming used by propagators and FWI;
   - plotting orientation, especially transpose in waveform 2D plots.

## Optimization Path

The survey optimization should improve definitions and contracts first, then
only make behavior changes when validation can prove they are safe.

The guiding survey lifecycle is:

```text
source/receiver definitions
-> Survey geometry and optional masks
-> propagator consumes source/receiver tensors
-> SeismicData records waveform dictionaries
-> save/load observed or synthetic data
-> FWI consumes SeismicData
```

### Phase 1 - Responsibility And Contract Audit

Goal:
document the actual ownership of source, receiver, survey, masks, and seismic
data.

Validation:
documentation review plus `py_compile` if files are touched.

Stop:
ownership map is clear enough to guide code changes.

### Phase 2 - Low-Risk Readability Cleanup

Potential targets:

- concise module docstrings;
- remove duplicate imports;
- clarify method docstrings and error messages;
- replace broad `except` in `__repr__` only if a focused smoke test covers empty
  and non-empty objects;
- add small tests for source/receiver shape contracts.

Validation:

- `conda run -n adfwi python -m py_compile ADFWI/survey/*.py`;
- focused `Source`/`Receiver`/`Survey`/`SeismicData` tests;
- Marmousi2 validation `check` stage if a caller contract is touched.

Numerical comparison:
not required if only comments/imports/docstrings change.

### Phase 3 - Data Contract Tests

Potential targets:

- `SeismicData.record_data()` conversion behavior;
- `save()`/`load()` round trip;
- acoustic and elastic parse functions;
- normalization behavior for zero traces.

Validation:

- small deterministic arrays for acoustic and elastic data;
- save/load `.npz` round trip;
- exact `max_abs=0` comparison for stored arrays.

Numerical comparison:
required if any waveform data content or dtype can change.

### Phase 4 - Receiver Mask Contract

Potential targets:

- `Survey.set_receiver_masks()`;
- `receiver_masks_obs` ownership and downstream expectations;
- plotting masked single-shot receiver geometry.

Validation:

- mask shape/value tests;
- existing backend integration receiver-mask tests;
- a small forward comparison if mask application reaches propagator/FWI.

Numerical comparison:
required if mask behavior changes propagator or FWI inputs.

## First Recommended Task

Start with Phase 1 and a small part of Phase 2 only.

Suggested first bounded round:

```text
Goal:
Clarify ADFWI/survey responsibility boundaries and clean stale comments/imports.

Scope:
ADFWI/survey/*.py and bv1.2-survey docs.

Validation:
py_compile plus focused source/receiver/survey construction smoke tests.

Stop:
No waveform data, receiver mask, save/load, or propagator-facing behavior changes.
```

## Validation Policy

Use the lightest validation that proves the change:

| Change | Validation |
| --- | --- |
| docs/comments/import cleanup | `py_compile`, import checks |
| source/receiver shape contract | focused unit tests |
| receiver mask validation | focused mask tests plus backend integration mask tests |
| `SeismicData.record_data` or parse behavior | exact array comparison |
| save/load format | round-trip `.npz` comparison |
| propagator-facing geometry changes | validation forward output comparison |

Full Marmousi2 inversion is not needed for survey-layer readability changes.
