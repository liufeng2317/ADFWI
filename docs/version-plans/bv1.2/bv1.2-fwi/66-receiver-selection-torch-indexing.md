# 66 - Receiver Selection Torch Indexing

## Optimization Path

Continue from the receiver-selection and FWI data-contract cleanup by removing
the NumPy/CPU indexing step from trace-missing receiver selection.

The goal is to keep synthetic/observed receiver matching on the active torch
device when observed data contains fewer receiver traces than the synthetic
record.

## Change

- select_or_mask_receivers now converts receiver masks directly on
  synthetic.device and uses torch.nonzero plus index_select for active
  receiver gathering.
- Same-shape synthetic/observed data still uses the existing broadcast mask
  multiplication path.
- Added explicit validation for receiver-mask dimensionality, shot-count
  mismatch, and active-receiver count mismatch.
- Extended receiver-selection tests for non-float/list masks and mismatch
  diagnostics.

## Scientific Contract

- No misfit formula, transform order, propagator output, or FWI optimizer logic
  changed.
- The selected receiver traces preserve the legacy active-receiver order.
- Existing receiver-selection comparison tests verify exact tensor equality
  against the legacy NumPy ordering for same-shape masking and trace-missing
  selection.
- Because this change affects the pre-loss FWI data path, exact receiver-selection
  equality is the numerical guard for this step. Full FWI smoke is not required
  unless this helper is later wired into a broader data-path change.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python -m unittest tests/test_receiver_selection.py: 6 tests passed.
- conda run -n adfwi python -m unittest tests/test_fwi_data_contract.py: 27 tests passed.
- conda run -n adfwi python -m py_compile ADFWI/fwi/transforms/receivers.py: passed.

The receiver-selection tests include exact tensor equality against the legacy
selection behavior for same-shape masking and trace-missing selection.

## Next Optimization Direction

Add benchmark scaffolding before larger performance work. The next practical
step is a small backend-aware benchmark runner that records acoustic forward
runtime, one-iteration inversion runtime, accelerator memory, gradient norm,
loss, and backend diagnostics for CPU/CUDA/NPU comparisons.
