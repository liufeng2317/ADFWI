# 120 - Gradient Parameter Ownership

## Optimization Path

Continue the convergence cleanup around `ADFWI.fwi.runtime.gradient`. The
runtime gradient module should own generic gradient processor dispatch, while
physical parameter order belongs to the concrete FWI drivers that understand
acoustic and elastic model semantics.

This is the same ownership rule used for elastic loss components: the driver
chooses the physical fields; shared helpers only execute generic mechanics.

## Change

- Removed acoustic/elastic parameter-order constants and convenience helpers
  from `ADFWI.fwi.runtime.gradient`.
- Kept `parameter_specs(...)` in `runtime.gradient` because it is a generic
  `(name, idx)` adapter for legacy list-style gradient processors.
- Added acoustic parameter ownership to `ADFWI.fwi.acoustic_fwi`:
  - `ACOUSTIC_PARAMETER_NAMES`;
  - `acoustic_parameter_names()`;
  - `acoustic_gradient_parameter_specs()`.
- Added elastic parameter ownership to `ADFWI.fwi.elastic_fwi`:
  - `ELASTIC_ISOTROPIC_PARAMETER_NAMES`;
  - `ELASTIC_ANISOTROPIC_PARAMETER_NAMES`;
  - `ELASTIC_ANISOTROPIC_GRADIENT_NAMES`;
  - `elastic_parameter_names(...)`;
  - `elastic_gradient_parameter_specs(...)`.
- Updated runtime tests to validate the parameter contracts from the owning
  drivers instead of from the runtime gradient module.
- Updated runtime package documentation to describe `runtime.gradient` as
  gradient processor dispatch rather than parameter-spec ownership.

## Scientific Contract

- Acoustic parameter order remains `vp`, `rho`.
- Acoustic gradient processor indices remain `vp -> 0`, `rho -> 1`.
- Isotropic elastic parameter order remains `vp`, `vs`, `rho`.
- Anisotropic elastic regularization/cache order remains `vp`, `vs`, `rho`,
  `eps`, `delta`, `gamma`.
- Anisotropic elastic gradient processor specs remain `vp -> 0`, `vs -> 1`,
  `rho -> 2`, `eps -> 3`, `delta -> 4`; `gamma` remains excluded from the
  current ElasticFWI gradient processor path.
- No loss, forward, backward, regularization, cache, optimizer, or numerical
  gradient formula changed.

## Validation

Completed validation:

- `python -m py_compile ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/__init__.py ADFWI/fwi/acoustic_fwi.py ADFWI/fwi/elastic_fwi.py tests/test_fwi_runtime.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_fwi_iteration.py tests/test_backend_integration.py tests/test_import_surface_policy.py`: passed.
- `git diff --check`: passed.

## Next Optimization Direction

Do not continue broad module reshuffling. The next useful convergence step is a
small runtime docstring/API pass that verifies `backend`, `forward`, `wavefield`,
`regularization`, `gradient`, and `cache` each describe one clear responsibility
and that active imports match those owner modules.
