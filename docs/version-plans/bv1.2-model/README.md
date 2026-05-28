# bv1.2 Model Optimization

This folder tracks the bounded optimization plan for `ADFWI/model`.

The goal is to make model definitions easier to understand and safer to use
without changing numerical behavior unless a specific numerical validation is
planned.

## Read First

- [00 - Model Optimization Outline](./00-model-optimization-outline.md)
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

## Stop Rule

Stop this stage when model responsibilities, parameter lifecycle, and validation
contracts are clear. Do not continue into propagator or example migration as
part of this stage.
