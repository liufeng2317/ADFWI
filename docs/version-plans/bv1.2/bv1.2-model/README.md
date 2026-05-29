# bv1.2 Model Optimization

This folder tracks the bounded optimization plan for `ADFWI/model`.

The goal is to make model definitions easier to understand and safer to use
without changing numerical behavior unless a specific numerical validation is
planned.

## Read First

- [00 - Model Optimization Outline](./00-model-optimization-outline.md)
- [01 - Model Definition Readability](./01-model-definition-readability.md)
- [02 - Model Mechanics Convergence](./02-model-mechanics-convergence.md)
- [03 - Parameter Formula Semantics Audit](./03-parameter-formula-semantics-audit.md)
- [04 - TI Anisotropic Type Validation](./04-ti-anisotropic-type-validation.md)
- [05 - Staggered Grid Validation](./05-staggered-grid-validation.md)
- [06 - Model Closeout Summary](./06-model-closeout-summary.md)
- [07 - Model Closeout Real Case Validation](./07-model-closeout-real-case-validation.md)
- [Model Optimization Map](./model-optimization-map.md)
- [Global Optimization Skill](../optimization-skill.md)

## Current Boundary

Focus only on:

- `ADFWI/model/base.py`
- `ADFWI/model/acoustic_model.py`
- `ADFWI/model/elastic_model.py`
- `ADFWI/model/parameters.py`

Do not include `ADFWI/propagator`, `ADFWI/fwi`, `ADFWI/survey`, or examples in
this stage unless a model-layer change requires a focused validation caller.

## Stage Status

This model-stage optimization is closed. Continue only for a clearly scoped
follow-up with its own validation target.

## Stop Rule

Stop this stage when model responsibilities, parameter lifecycle, and validation
contracts are clear. Do not continue into propagator or example migration as
part of this stage.
