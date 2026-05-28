# 134 - FWI Package Readability Closeout

## Optimization Path

After the Marmousi2 validation scripts and notebooks were aligned, the next
closeout pass focused on `ADFWI/fwi` readability rather than new behavior. The
goal was to reduce unclear package entry points and obvious local clutter
without changing FWI numerical paths.

## Audit Findings

- `ADFWI/fwi/__init__.py` exposed the two FWI drivers but had no package-level
  explanation or explicit public API list.
- `ADFWI/fwi/misfit/__init__.py` and `ADFWI/fwi/regularization/__init__.py`
  exported historical names without explaining that these names are the public
  compatibility surface.
- `Misfit` and `Regularization` used `abstractmethod` but did not inherit from
  `ABC`, so they read like abstract bases without enforcing that contract.
- Several FWI driver figure-saving helpers used `if ...: pass` branches for
  disabled output, which made the no-op path harder to read.
- Generated `__pycache__` directories and untracked `backup` directories existed
  under `ADFWI/fwi`, creating noise during package-structure audits. They were
  not tracked by git.

## Change

- Added a top-level `ADFWI.fwi` package docstring and `__all__`.
- Added explicit public `__all__` lists for `ADFWI.fwi.misfit` and
  `ADFWI.fwi.regularization`.
- Converted `Misfit` and `Regularization` into real `ABC` abstract base
  classes.
- Replaced empty abstract `pass` bodies with `raise NotImplementedError`.
- Clarified `regular_StepLR` naming internally without changing its public
  function name.
- Replaced figure-saving no-op branches in acoustic/elastic FWI drivers with
  early returns.
- Removed untracked generated/cache clutter from the local `ADFWI/fwi` tree.

## Scientific Contract

- No FWI driver loop, propagator call, loss formula, gradient processor, or
  transform behavior changed.
- The public historical class/function names remain available.
- This is a readability/API-surface closeout, not a numerical optimization.

## Validation

Completed validation:

- `conda run -n adfwi python -m py_compile $(git ls-files 'ADFWI/fwi/*.py' 'ADFWI/fwi/*/*.py')`: passed.
- Public import check for `AcousticFWI`, `ElasticFWI`, misfit classes, and
  regularization classes: passed.
- Abstract base instantiation checks for `Misfit` and `Regularization`: passed.
- `conda run -n adfwi python -m unittest tests/test_fwi_iteration_loss.py tests/test_misfit_squared_l2.py tests/test_backend_integration.py`: passed.

## Next Direction

Do not continue broad reshuffling. Further `ADFWI/fwi` work should be limited
to specific findings with tests, such as documenting optimizer legacy behavior
or adding targeted comments in large driver methods.
