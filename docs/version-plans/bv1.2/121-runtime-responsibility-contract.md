# 121 - Runtime Responsibility Contract

## Optimization Path

Continue the convergence-focused cleanup after moving gradient parameter
ownership into the FWI drivers. The next risk in `ADFWI.fwi.runtime` was not
execution behavior, but unclear module boundaries: the package could still be
mistaken for a full forward/inversion framework instead of a set of small driver
runtime helpers.

This pass therefore updates only responsibility documentation and keeps all
runtime function bodies unchanged.

## Change

- Clarified `ADFWI.fwi.runtime.__init__` as a namespace of shared execution
  mechanics, not a high-level forward or inversion framework.
- Clarified module ownership:
  - `backend`: construction-time backend/device validation and regularization
    alignment;
  - `forward`: one-batch propagator execution records;
  - `wavefield`: forward-wavefield extraction/accumulation for gradient
    processors;
  - `regularization`: shared model-space regularization summation mechanics;
  - `gradient`: gradient processor dispatch and `parameter.grad` write-back;
  - `cache`: inversion-history bookkeeping.
- Explicitly documented that physical choices remain in `AcousticFWI` and
  `ElasticFWI`: model parameters, inversion components, loss components, and
  regularization weights.

## Scientific Contract

- No function body changed.
- No tensor operation, forward propagation, loss construction, backward pass,
  regularization value, gradient processor formula, cache value, or optimizer
  step changed.
- This is a readability and ownership-contract pass only.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile ADFWI/fwi/runtime/__init__.py ADFWI/fwi/runtime/backend.py ADFWI/fwi/runtime/cache.py ADFWI/fwi/runtime/forward.py ADFWI/fwi/runtime/gradient.py ADFWI/fwi/runtime/regularization.py ADFWI/fwi/runtime/wavefield.py`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_runtime.py tests/test_import_surface_policy.py`: passed.
- `git diff --check`: passed.

## Next Optimization Direction

Stop runtime reshuffling here unless a concrete bug or duplicate execution path
appears. The next useful convergence work is to inspect `AcousticFWI` and
`ElasticFWI` themselves for small owner-level constants or helper names that can
be clarified without moving numerical logic.
