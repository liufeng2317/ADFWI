# 00 - Model Optimization Outline

## Goal

Improve the readability and robustness of `ADFWI/model` while preserving model
numerical behavior by default.

The model layer should clearly answer:

- what is a persistent model parameter;
- what is a derived physical quantity;
- when model parameters are converted to `torch.nn.Parameter`;
- when bounds, water-layer masks, and empirical updates are applied;
- which formulas are acoustic/elastic physics and therefore require numerical
  validation if changed.

## Scope

Allowed files for the first model stage:

- `ADFWI/model/base.py`
- `ADFWI/model/acoustic_model.py`
- `ADFWI/model/elastic_model.py`
- `ADFWI/model/parameters.py`
- focused tests under `tests/` if needed
- records under `docs/version-plans/bv1.2-model/`

Non-scope:

- propagator kernels;
- FWI loop behavior;
- survey/source/receiver definitions;
- example migration;
- DIP model wrappers.

## Current Observations

The current model layer is compact, but the responsibility boundaries are not
fully explicit:

- `AbstractModel` owns geometry, backend placement, bounds dictionaries,
  parameter access, and generic clipping/checking helpers.
- `AcousticModel` owns `vp/rho`, empirical `vp <-> rho` updates, water-layer
  preservation, plotting helpers, and acoustic forward-time model constraints.
- `IsotropicElasticModel` and `AnisotropicElasticModel` own `vp/vs/rho` and
  optional Thomsen parameters, but repeat several acoustic patterns.
- `parameters.py` owns physical parameter transforms and staggered-grid helper
  formulas; changes here are numerical changes.

## Optimization Path

The model optimization should improve definitions and algorithmic readability,
not add structure for its own sake. Prefer in-place clarification and thin
shared helpers over new files or deeper abstractions.

The guiding model lifecycle is:

```text
input arrays
-> persistent model parameters
-> optional empirical updates
-> bounds and water-layer constraints
-> derived physical quantities
-> propagator/FWI consumers
```

### Phase 1 - Responsibility And Contract Audit

Goal:
document the actual ownership of geometry, trainable parameters, bounds,
derived quantities, empirical updates, and plotting helpers.

Validation:
documentation/link checks only.

Stop:
model ownership map is clear enough to guide code changes.

### Phase 2 - Low-Risk Readability Cleanup

Potential targets:

- replace stale header comments with concise module docstrings;
- clarify docstrings that say "Non use" or mention only CPU/CUDA;
- remove unused imports if confirmed;
- make helper names/comments describe parameter lifecycle more clearly;
- clarify that `forward()` updates constraints and derived model quantities
  rather than running wave propagation.
- avoid splitting files or introducing new abstractions in this phase.

Validation:

- `python -m py_compile ADFWI/model/*.py`;
- focused model import/constructor checks;
- existing backend/model smoke tests if available.

Numerical comparison:
not required if only comments/imports/docstrings change.

### Phase 3 - Shared Model Mechanics

Potential targets:

- common helper for registering model parameter tensors;
- common helper for bounds initialization;
- common helper for water-layer-preserving clamp;
- common helper for empirical update replacement of `rho` or `vp`.

Only extract helpers that remove real duplication and have stable semantics.
Do not create a larger model framework.

Validation:

- compare acoustic and elastic small-model arrays before/after;
- check `requires_grad`, device, dtype, and bounds behavior;
- ensure water-layer mask preserves original values.

Numerical comparison:
required if helper extraction changes any tensor values, dtype/device behavior,
or `requires_grad` behavior.

### Phase 4 - Formula-Sensitive Cleanup

Potential targets:

- `vs_vp_to_Lame`;
- `thomsen_to_elastic_moduli`;
- `elastic_moduli_to_thomsen`;
- `elastic_moduli_for_isotropic`;
- `elastic_moduli_for_TI`;
- `parameter_staggered_grid`.

Validation:

- focused tensor formula tests with hand-checkable or saved reference values;
- CPU/NPU or CPU/CUDA comparison when device behavior matters;
- max absolute and relative error report.

Numerical comparison:
always required.

## First Recommended Task

Start with Phase 1 and Phase 2 only.

Suggested first bounded round:

```text
Goal:
Clarify ADFWI/model responsibility boundaries and clean stale comments/imports.

Scope:
ADFWI/model/*.py and bv1.2-model docs.

Validation:
py_compile and import/constructor smoke checks.

Stop:
No formula, constraint, or tensor-value behavior changes.
```

## Validation Policy

Use the lightest validation that proves the change:

| Change | Validation |
| --- | --- |
| docs/comments/import cleanup | `py_compile`, imports |
| parameter registration helper | small acoustic/elastic model value comparison |
| bounds or water-layer logic | explicit clamp/mask tests |
| physical transform formulas | numerical precision tests with reported error |
| model-forward behavior | small forward output comparison |

Full Marmousi2 validation is not needed for model-layer readability changes.
