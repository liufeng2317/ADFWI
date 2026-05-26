# 34. Core Comment and Contract Pass

## Goal

Review whether the bv1.2 structural optimization remains understandable for geophysical researchers. This pass improves comments, docstrings, variable definitions, and type hints around the shared helpers introduced during the FWI cleanup.

The intent is documentation clarity only. No numerical formula, tensor operation order, backend selection rule, or public FWI usage is changed.

## Scope

Updated core contract comments in:

- `ADFWI/backends/backend.py`
- `ADFWI/fwi/runtime/backend.py`
- `ADFWI/fwi/runtime/gradient.py`
- `ADFWI/fwi/runtime/regularization.py`
- `ADFWI/fwi/data/preparation.py`
- `ADFWI/fwi/iteration/range.py`

## Clarified Concepts

| Area | Clarification |
| --- | --- |
| Backend tensor factories | Explicitly states that helper-created tensors use backend dtype/device defaults. |
| Backend preference | Defines `prefer` as either comma-separated string or iterable of backend family names. |
| Backend resolution | Defines explicit vs automatic device resolution and fallback diagnostics. |
| Model/propagator device guard | Explains why model and propagator devices must match for differentiable forward modeling. |
| Regularization backend alignment | Defines in-place movement of regularization metadata and tensor attributes. |
| Runtime gradient processing | Defines `parameter`, `gradient_processor`, `forw`, `idx`, and the legacy CPU NumPy processor contract. |
| Transform context | Defines `shot_index`, `cutoff_freq`, `dt`, mute parameters, geometry arrays, receiver masks, and data masks. |
| Loss pair preparation | States the normal waveform shape `[shot, time, receiver]` and explains receiver selection before same-shape transforms. |
| Batch ranges | Defines `batch`, half-open `[begin, end)` shot interval, and `shot_index` usage. |

## Design Rule Confirmed

The comments now reinforce the current module boundary:

- `acoustic_fwi.py` and `elastic_fwi.py` remain readable inversion drivers.
- `runtime` helpers are shared execution utilities, not new physical operators.
- `data` and `transforms` define the pre-misfit waveform contract.
- `iteration` defines loop bookkeeping, not propagation physics.

## Validation

Because this pass changes documentation strings and type annotations only, validation focuses on syntax and import safety:

- `python -m py_compile ADFWI/backends/backend.py ADFWI/fwi/runtime/backend.py ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/regularization.py ADFWI/fwi/data/preparation.py ADFWI/fwi/iteration/range.py`: passed.
- `python -m unittest tests/test_backends.py tests/test_fwi_runtime.py tests/test_fwi_data_contract.py tests/test_fwi_iteration.py`: 54 tests passed, 1 skipped.

No smoke comparison is required because no executable numerical path was changed.
