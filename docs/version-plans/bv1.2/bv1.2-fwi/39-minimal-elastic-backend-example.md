# 39. Minimal Elastic Backend Example

## Goal

Add a script-style isotropic elastic FWI example that mirrors the minimal acoustic backend example. This demonstrates that the same bv1.2 `ADFWI.set_backend(...)` pattern works for ElasticFWI.

## Scope

Added `scripts/examples/minimal_elastic_fwi_backend.py`.

The script performs the following in memory:

1. configure backend with `ADFWI.set_backend(...)`;
2. build one tiny true isotropic elastic model and one initial elastic model;
3. build one source and three pressure receivers;
4. synthesize observed elastic data from the true model;
5. run one pressure-component `ElasticFWI` iteration with L2 misfit;
6. print a JSON summary containing backend diagnostics, loss, gradient norm, and model update norm.

## Boundary

The example writes no files, figures, notebooks, wavefields, or example outputs. It does not introduce new FWI behavior; it reuses the existing ElasticFWI path and backend API.

## Usage

```bash
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device npu:0
```

## Validation

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile scripts/examples/minimal_elastic_fwi_backend.py`: passed.
- `conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device cpu`: passed; loss `0.0008452539914287627`, `vp_grad_norm=4.3503572669578716e-05`, `vp_update_norm=4.350393295288086`.
- `conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device npu:0`: passed; loss `0.0008452520705759525`, `vp_grad_norm=4.350356903159991e-05`, `vp_update_norm=4.350393295288086`.

The CPU and NPU runs both returned finite nonzero gradients and model updates while writing no output files.
