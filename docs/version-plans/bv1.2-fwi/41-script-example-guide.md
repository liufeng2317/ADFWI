# Script Example Guide

## Purpose

The backend work now has minimal acoustic and elastic FWI scripts plus a smoke runner that validates them on CPU/NPU. This pass adds a researcher-facing guide at `scripts/examples/README.md` so the examples are easier to discover, run, and adapt.

## What Changed

- Added `scripts/examples/README.md`.
- Linked the guide from `docs/backend-usage.md`.
- Documented the intended use of the minimal acoustic and elastic examples.
- Recorded the one-line backend pattern for `ADFWI.set_backend(...)`.
- Added a migration table showing which synthetic blocks should be replaced when moving toward real cases such as Marmousi2.

## User Contract

The script examples are intended to remain safe entry points for backend and FWI workflow checks:

- they should not write notebooks, figures, wavefields, or data files by default;
- they should use one top-level backend setup call before creating ADFWI objects;
- they should print JSON summaries that include device, dtype, loss, gradient norm, and update norm;
- they should be runnable from the repository root with `conda run -n adfwi python ...`.

## Validation

This is a documentation and usability pass. Validation should confirm that:

```bash
python -m py_compile scripts/examples/minimal_acoustic_fwi_backend.py scripts/examples/minimal_elastic_fwi_backend.py scripts/smoke/run_backend_smoke_suite.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic
```

The existing numerical contract remains the one pinned by the examples smoke suite: acoustic and elastic CPU/NPU runs must complete with finite positive loss, nonzero `vp` gradient norm, nonzero model update norm, and CPU-vs-NPU metric differences below the configured tolerance.

Local validation after adding the guide reported `runs: 4`, `ok: 4`, `failed: 0`, `max_abs_diff: 1.1920928955078125e-07`, and `max_rel_diff: 2.2725155155996827e-06`.

## Next Step

The next natural step is to create a script-style Marmousi2 acoustic entry point that follows this guide while leaving the existing notebooks untouched. That script can become the first bridge between tiny backend examples and full research cases.
