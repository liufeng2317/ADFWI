# bv1.2 Utils Optimization

This folder tracks bounded cleanup for `ADFWI/utils`.

The goal is not to split utilities aggressively. The goal is to make the
current helper layer understandable: what belongs here, what is legacy example
support, what has numerical behavior, and what must be validated before any
change.

## Read First

- [00 - Utils Optimization Outline](./00-utils-optimization-outline.md)
- [01 - Utils Planning Audit](./01-utils-planning-audit.md)
- [02 - Utils Public Namespace Readability](./02-utils-public-namespace-readability.md)
- [Utils Optimization Map](./utils-optimization-map.md)
- [Global Optimization Skill](../optimization-skill.md)

## Current Boundary

`ADFWI/utils` is a shared helper layer used by examples, survey construction,
FWI transforms, propagators, benchmarks, and validation scripts.

It currently contains:

- array/tensor conversion helpers;
- source wavelet generation;
- benchmark model loading/resampling/smoothing helpers;
- legacy mute and first-arrival helpers used by transform parity tests;
- simple signal-processing and plotting helpers;
- simple model-quality metrics;
- noise injection helpers.

## Stop Rule

Each utils cleanup round must have one bounded goal and a direct validation
command. Do not rewrite numerical helper behavior without a before/after
comparison of outputs, shapes, dtypes, and finite values.
