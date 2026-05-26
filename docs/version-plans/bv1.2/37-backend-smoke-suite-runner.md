# 37. Backend Smoke Suite Runner

## Goal

Add a layered smoke runner for new CPU/NPU/CUDA machines. The runner groups lightweight public API checks, tensor-level misfit checks, forward propagation checks, and mini-inversion comparison checks behind one JSON-producing command.

## Scope

Added `scripts/smoke/run_backend_smoke_suite.py`.

Supported suites:

| Suite | Meaning | Underlying script |
| --- | --- | --- |
| `public` | Top-level backend API and diagnostics | `backend_public_api_smoke.py` |
| `misfit` | Tensor-level misfit loss/backward checks | `misfit_backend_smoke.py` |
| `acoustic-forward` | Acoustic forward/backward smoke | `acoustic_backend_smoke.py` |
| `elastic-forward` | Elastic forward/backward smoke | `elastic_backend_smoke.py` |
| `compare-mini` | CPU/device mini-inversion drift comparison | `compare_backend_smoke.py` |

## Usage

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public --devices cpu,npu:0
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,misfit --devices cpu,npu:0 --misfits L2,StudentT
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,compare-mini --devices cpu,npu:0 --compare-cases trace-missing
```

## Boundary

This is orchestration only. It does not change any backend, propagator, transform, misfit, or FWI numerical implementation.

## Validation

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile scripts/smoke/run_backend_smoke_suite.py`: passed.
- `conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public --devices cpu,npu:0`: passed; CPU and NPU public API checks returned `ok`.
- `conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,misfit --devices cpu,npu:0 --misfits L2,StudentT`: passed; public and tensor-level misfit checks returned `ok` on CPU and NPU.

The heavier `compare-mini` suite is wired through the existing `compare_backend_smoke.py` path and remains opt-in because it runs mini inversion comparisons.
