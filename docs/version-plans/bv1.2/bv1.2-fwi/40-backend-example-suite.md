# Backend Example Suite

## Purpose

The bv1.2 backend work now has two researcher-facing minimal scripts:

- `scripts/examples/minimal_acoustic_fwi_backend.py`
- `scripts/examples/minimal_elastic_fwi_backend.py`

This pass connects those scripts to `scripts/smoke/run_backend_smoke_suite.py` through a new `examples` suite. The goal is to make the public examples part of the same CPU/NPU validation path as the backend API, misfit, forward, and mini-inversion checks.

## Implementation

`run_backend_smoke_suite.py` now supports:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic
```

The suite runs each selected minimal example once per requested device. Each child script still writes no notebooks, figures, wavefields, or example outputs; the runner only captures each JSON summary and reports a single combined JSON result.

When a CPU reference run is present, the suite also compares each non-reference device against CPU for `loss`, `vp_grad_norm`, and `vp_update_norm`. The default tolerances are `--example-rtol 1e-4` and `--example-atol 1e-8`.

The runner also adds a top-level `summary` block. The detailed `reports` payload remains unchanged, while `summary.by_suite` gives compact counts and, for the `examples` suite, the maximum absolute and relative CPU-vs-device metric differences.

## Scientific Contract

The example suite checks that the public one-line backend API is usable through complete tiny FWI workflows:

- acoustic example: synthesize pressure observations, run one `AcousticFWI` iteration, and verify finite positive loss, `vp` gradient norm, and model update norm;
- elastic example: synthesize elastic observations, run one pressure-component `ElasticFWI` iteration, and verify finite positive loss, `vp` gradient norm, and model update norm;
- CPU and NPU requests use the same tiny model geometry, survey, seed, dtype, and checkpoint segment count;
- no persistent example artifacts are created.

## Validation

Validation commands for this pass:

```bash
python -m py_compile scripts/smoke/run_backend_smoke_suite.py scripts/examples/minimal_acoustic_fwi_backend.py scripts/examples/minimal_elastic_fwi_backend.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic
```

The expected result is a top-level JSON object with `status: ok`, containing four successful runs: acoustic CPU, acoustic NPU, elastic CPU, and elastic NPU. The exact timing values are machine-dependent and are not part of the numerical contract.

Observed CPU-vs-NPU comparison on the local `adfwi` environment:

| Problem | Metric | CPU reference | NPU value | Abs diff | Rel diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| acoustic | loss | `3.8032811744415085e-07` | `3.8032811744415085e-07` | `0.0` | `0.0` |
| acoustic | `vp_grad_norm` | `1.8180083216634557e-08` | `1.8180081440277718e-08` | `1.7763568394002505e-15` | `9.77089498564509e-08` |
| acoustic | `vp_update_norm` | `1.8180636167526245` | `1.818063735961914` | `1.1920928955078125e-07` | `6.556937196934264e-08` |
| elastic | loss | `0.0008452539914287627` | `0.0008452520705759525` | `1.9208528101444244e-09` | `2.2725155155996827e-06` |
| elastic | `vp_grad_norm` | `4.3503572669578716e-05` | `4.350356903159991e-05` | `3.637978807091713e-12` | `8.362482857955452e-08` |
| elastic | `vp_update_norm` | `4.350393295288086` | `4.350393295288086` | `0.0` | `0.0` |

All observed relative differences are below the default `1e-4` relative tolerance. The top-level `summary.by_suite` entry reports `max_abs_diff` and `max_rel_diff` so future runs can be checked quickly without manually scanning every child JSON payload.

For the local validation run after adding `summary`, the examples suite reported `runs: 4`, `ok: 4`, `failed: 0`, `max_abs_diff: 1.1920928955078125e-07`, and `max_rel_diff: 2.2725155155996827e-06`.

## Notes

This change does not modify propagation kernels, FWI loss construction, gradient processing, model updates, or transform behavior. It only promotes the existing minimal examples into the layered smoke runner so future backend refactors can validate the documented user path with one command.
