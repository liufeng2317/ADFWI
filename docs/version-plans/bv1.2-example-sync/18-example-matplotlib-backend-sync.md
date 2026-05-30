# 18 Example Matplotlib Backend Sync

## Scope

Target:

```text
examples/**/*.py
```

This round only removed hard-coded Matplotlib backend selection from example
Python scripts. It did not change plotting calls, saved figure names, inversion
parameters, backend/device setup, or numerical logic.

## Problem

Many scripts contained:

```python
import matplotlib
matplotlib.use("agg")
```

This forces a non-interactive backend from inside examples and can conflict
with local notebooks, IDEs, display servers, or user-selected Matplotlib
configuration. In several scripts it was also placed after importing
`matplotlib.pyplot`, which is not a clean backend-selection contract.

## Change

- Removed `matplotlib.use("agg")` from example `.py` files.
- Removed the now-unused bare `import matplotlib`.
- Kept `import matplotlib.pyplot as plt` unchanged where scripts already used
  pyplot.

Users who need headless execution can still choose a backend outside the script,
for example with environment configuration or a wrapper command.

## Validation

- Residual scan found no `matplotlib.use("agg")` or unused bare
  `import matplotlib` in `examples/**/*.py`.
- `py_compile` passed for all scripts touched by this backend cleanup.

## Next Direction

Continue with smaller readability passes only where they remove concrete
confusion. The remaining common issue is duplicated local imports inside DR-FWI
scripts, which should be handled separately from backend behavior.
