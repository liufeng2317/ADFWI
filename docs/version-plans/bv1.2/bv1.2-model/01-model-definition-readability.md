# 01 - Model Definition Readability

## Goal

Clarify `ADFWI/model` definitions without changing model values, formulas,
constraints, or update order.

## Scope

- `ADFWI/model/__init__.py`
- `ADFWI/model/base.py`
- `ADFWI/model/acoustic_model.py`
- `ADFWI/model/elastic_model.py`
- `ADFWI/model/parameters.py`

## Change

This first implementation round is intentionally conservative:

- replace stale file headers with short module docstrings;
- make the model package public API explicit;
- make `AbstractModel` read as a real abstract base class;
- remove imports that are not used by the model modules;
- clarify that model `forward()` refreshes model state for the propagator,
  rather than performing wave propagation.

No physical formulas or model update ordering should change.

## Validation

Completed validation:

- compile all `ADFWI/model/*.py` files;
- import public model classes;
- construct small acoustic/isotropic/anisotropic models;
- run the existing backend integration tests that construct acoustic and
  isotropic elastic models through active workflow code.

No numerical comparison was required because this round did not change physical
formulas, model update order, clipping semantics, or tensor values.

## Stop

Stop after definition readability is clear. Do not continue into helper
extraction, parameter formulas, propagator behavior, or example migration in
this round.
