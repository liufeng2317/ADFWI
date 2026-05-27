# 104. Example Import Surface Audit

## Goal

Audit examples, scripts, and generated API documentation for stale imports after
the bv1.2 import-surface cleanup.

## Optimization Path

1. Search examples, scripts, and sphinx API sources for removed or namespace-only
   surfaces:
   - `ADFWI.fwi.multiScaleProcessing`
   - `ADFWI.fwi.normalization`
   - `ADFWI.fwi.transforms.waveform`
   - package-level `ADFWI.fwi.iteration` imports
   - package-level `ADFWI.fwi.runtime` imports
2. Update the tracked elastic multi-scale example notebook from the removed
   `ADFWI.fwi.multiScaleProcessing` path to the canonical
   `ADFWI.fwi.multiscale` path.
3. Update the `LegacyLowPassFilter` docstring so it names the canonical
   low-pass owner module.

Ignored workspace files under `docs/sphinx/` and an untracked acoustic
multi-scale notebook were also scanned locally, but they are not added to this
commit because those paths are ignored/generated in the current repository.

## Files Updated

- `examples/multi-scale/Iso-elastic-Marmousi2-multifreq/01_forward.ipynb`
- `ADFWI/fwi/transforms/filters.py`

## Numerical Contract

The notebook and sphinx changes are import-path updates only. The canonical
`ADFWI.fwi.multiscale` module re-exports the same legacy low-pass implementation
from `ADFWI.fwi.multiscale.legacy_lowpass`, so low-pass numerics are unchanged.

## Validation Result

Tracked notebook JSON validation passed:

```bash
python - <<'PY'
import json
for path in [
    "examples/multi-scale/Iso-elastic-Marmousi2-multifreq/01_forward.ipynb",
]:
    json.load(open(path))
    print(path, "OK")
PY
# examples/multi-scale/Iso-elastic-Marmousi2-multifreq/01_forward.ipynb OK
```

Stale imports are gone from tracked non-history sources:

```bash
git grep -n "multiScaleProcessing\\|ADFWI\\.fwi\\.normalization\\|transforms\\.waveform" -- ADFWI examples scripts docs tests README.md ':!docs/version-plans/bv1.2' ':!docs/version-plans/v1.1'
# ADFWI/fwi/multiscale/legacy_lowpass.py:4 keeps one historical ownership note.

git grep -n "from ADFWI\\.fwi\\.(iteration\\|runtime) import\\|ADFWI\\.fwi\\.(iteration\\|runtime) import" -- ADFWI examples scripts docs tests README.md ':!docs/version-plans/bv1.2' ':!docs/version-plans/v1.1'
# no matches
```

Focused multiscale/low-pass tests passed:

```bash
conda run -n adfwi python -m unittest tests/test_multiscale_compat.py tests/test_lowpass_transform_comparison.py
# Ran 7 tests in 0.056s, OK

conda run -n adfwi python -m py_compile ADFWI/fwi/transforms/filters.py ADFWI/fwi/multiscale/__init__.py ADFWI/fwi/multiscale/legacy_lowpass.py
# OK

conda run -n adfwi python -c 'from ADFWI.fwi.multiscale import lpass, lowpass, adj_lowpass; print(lpass.__name__, lowpass.__name__, adj_lowpass.__name__)'
# lpass lowpass adj_lowpass
```

## Next Direction

Continue auditing example notebooks for stale comments that describe removed
compatibility shims. Avoid executing heavy scientific notebooks during this
cleanup unless a changed cell needs runtime validation.
