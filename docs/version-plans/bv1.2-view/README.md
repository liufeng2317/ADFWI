# bv1.2 View Optimization

This folder tracks bounded cleanup for `ADFWI/view`.

The view module should remain a plotting/output helper layer. It should not own
model definitions, survey state, waveform processing, FWI loss logic, or
numerical transforms.

## Read First

- [00 - View Optimization Outline](./00-view-optimization-outline.md)
- [View Optimization Map](./view-optimization-map.md)
- [Global Optimization Skill](../optimization-skill.md)

## Current Boundary

`ADFWI/view` owns:

- converting array/tensor inputs to CPU plotting arrays;
- drawing model, survey, waveform, boundary, and inversion-summary figures;
- saving figures when `save_path` is provided;
- closing figures when `show=False`.

It does not own:

- changing physical units or model arrays;
- waveform normalization policies used by FWI;
- survey geometry selection;
- inversion state or loss computation;
- backend/device policy.

## Stop Rule

Each view cleanup round must be visible, bounded, and validated by either
`py_compile`, focused plotting smoke tests with the `Agg` backend, or the
Marmousi2 validation case when a validation script depends on the changed
plotting path.
