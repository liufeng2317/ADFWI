# bv1.2 Propagator Optimization

This folder tracks the bounded optimization plan for `ADFWI/propagator`.

The goal is to make propagator responsibilities and contracts explicit before
touching numerical kernels. This stage starts with documentation only: no
kernel rewrite, no finite-difference formula change, and no FWI behavior change.

## Read First

- [00 - Propagator Optimization Outline](./00-propagator-optimization-outline.md)
- [01 - Propagator Wrapper Readability](./01-propagator-wrapper-readability.md)
- [02 - Propagator Wrapper Contract Tests](./02-propagator-wrapper-contract-tests.md)
- [03 - Boundary Profile Validation](./03-boundary-profile-validation.md)
- [04 - Boundary Condition Readability](./04-boundary-condition-readability.md)
- [05 - Non Kernel Cleanup Closeout](./05-non-kernel-cleanup-closeout.md)
- [Propagator Optimization Map](./propagator-optimization-map.md)
- [Global Optimization Skill](../optimization-skill.md)

## Current Boundary

Focus only on understanding and documenting:

- `ADFWI/propagator/acoustic_propagator.py`
- `ADFWI/propagator/elastic_propagator.py`
- `ADFWI/propagator/acoustic_kernels.py`
- `ADFWI/propagator/acoustic_kernels_bs.py`
- `ADFWI/propagator/elastic_kernels.py`
- `ADFWI/propagator/boundary_condition.py`
- `ADFWI/propagator/gradient_process.py`
- direct callers from `ADFWI/fwi`, `ADFWI/dip`, tests, and validation examples

Do not edit propagator kernels in the first round.

## Stop Rule

Stop this planning stage when ownership boundaries, risk levels, validation
requirements, and the first bounded implementation candidates are clear.
Do not continue into numerical edits without a dedicated validation plan.
