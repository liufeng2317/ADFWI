# bv1.2 Survey Optimization

This folder tracks the bounded optimization plan for `ADFWI/survey`.

The goal is to make acquisition geometry, source/receiver metadata, receiver
masks, and seismic-data storage contracts easier to understand without changing
forward/inversion numerical behavior by default.

## Read First

- [00 - Survey Optimization Outline](./00-survey-optimization-outline.md)
- [Survey Optimization Map](./survey-optimization-map.md)
- [Global Optimization Skill](../optimization-skill.md)

## Current Boundary

Focus only on:

- `ADFWI/survey/source.py`
- `ADFWI/survey/receiver.py`
- `ADFWI/survey/survey.py`
- `ADFWI/survey/data.py`
- focused tests under `tests/` if needed

Do not include `ADFWI/propagator`, `ADFWI/fwi`, `ADFWI/model`, or example
migration in this stage unless a survey-layer change requires a focused
validation caller.

## Stop Rule

Stop this stage when source/receiver/survey/data responsibilities and validation
contracts are clear. Do not continue into propagator or broad example migration
as part of this stage.
