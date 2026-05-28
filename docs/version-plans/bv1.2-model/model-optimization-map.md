# Model Optimization Map

This map shows the intended ownership boundaries for `ADFWI/model`.

## Layer Map

```mermaid
flowchart TD
    User[User / examples / FWI setup]
    User --> ModelAPI[ADFWI.model public API]

    ModelAPI --> Base[base.py\nAbstractModel]
    ModelAPI --> Acoustic[acoustic_model.py\nAcousticModel]
    ModelAPI --> Elastic[elastic_model.py\nIsotropicElasticModel\nAnisotropicElasticModel]
    ModelAPI --> Params[parameters.py\nphysical transforms]

    Base --> Geometry[geometry\nox oz nx nz dx dz]
    Base --> Backend[backend placement\ndevice dtype]
    Base --> Access[parameter access\nget/set model, bounds, grads]

    Acoustic --> AcousticParams[persistent parameters\nvp rho]
    Acoustic --> AcousticRules[constraints\nbounds, water mask,\nempirical vp/rho update]

    Elastic --> ElasticParams[persistent parameters\nvp vs rho eps gamma delta]
    Elastic --> ElasticDerived[derived quantities\nlambda, mu, buoyancy,\nelastic moduli, staggered grids]
    Elastic --> ElasticRules[constraints\nbounds, water mask,\nempirical vp/rho update]

    Params --> Formulas[formula-sensitive transforms\nLame, Thomsen, moduli,\nstaggered-grid averaging]

    AcousticParams --> Propagator[ADFWI.propagator\nwave simulation]
    ElasticDerived --> Propagator
    AcousticRules --> FWI[ADFWI.fwi\ninversion loop]
    ElasticRules --> FWI
```

## Responsibility Table

| File | Should Own | Should Not Own |
| --- | --- | --- |
| `base.py` | common geometry, backend placement, parameter access, bounds dictionaries, generic checks | acoustic/elastic physics formulas |
| `acoustic_model.py` | acoustic persistent parameters, empirical density/velocity update, acoustic constraints | wave propagation or FWI loss logic |
| `elastic_model.py` | elastic persistent parameters, anisotropic parameters, derived elastic quantities, elastic constraints | generic model helpers that can live in `base.py` |
| `parameters.py` | physical transform formulas and staggered-grid preparation | model object lifecycle or plotting |

## Parameter Lifecycle

```mermaid
flowchart LR
    Input[numpy/tensor input\nshape nz,nx]
    Input --> Validate[shape and bound checks]
    Validate --> Tensor[to backend tensor\ndevice dtype]
    Tensor --> Parameter[torch.nn.Parameter\nrequires_grad flag]
    Parameter --> Forward[model.forward()]
    Forward --> Empirical[optional empirical update]
    Forward --> Clamp[bounds + water mask]
    Clamp --> Derived[derived quantities\nelastic only]
    Derived --> Propagator[propagator inputs]
```

For acoustic models, `forward()` mainly applies optional empirical updates and
constraints. For elastic models, `forward()` also refreshes Lame parameters,
elastic moduli, and staggered-grid quantities.

## Risk Map

| Area | Risk | Validation Needed |
| --- | --- | --- |
| docstrings/imports/comments | low | compile/import checks |
| helper extraction with same tensor values | medium | small-model before/after comparison |
| bounds or water-layer mask behavior | medium | explicit mask/clamp tests |
| empirical `rho` or `vp` update | medium-high | saved value comparison |
| Thomsen/Lame/moduli/staggered formulas | high | numerical precision tests |
| propagator-facing output shapes | high | small forward simulation or existing propagator smoke |

## Preferred First Optimization

Start by clarifying names, comments, docstrings, and ownership without changing
formula outputs. That gives a stable reading path before any helper extraction
or numerical cleanup.
