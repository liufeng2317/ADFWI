# 38. Minimal Acoustic Backend Example

## Goal

Add a script-style user example that demonstrates the bv1.2 recommended backend pattern in a complete but tiny acoustic FWI workflow. The example is intended for researchers to copy into their own scripts without modifying notebook outputs.

## Scope

Added `scripts/examples/minimal_acoustic_fwi_backend.py`.

The script performs the following in memory:

1. configure backend with `ADFWI.set_backend(...)`;
2. build one tiny true acoustic model and one initial acoustic model;
3. build one source and three receivers;
4. synthesize observed data from the true model;
5. run one `AcousticFWI` iteration with L2 misfit;
6. print a JSON summary containing backend diagnostics, loss, gradient norm, and model update norm.

## Boundary

The example writes no files, figures, notebooks, wavefields, or example outputs. It does not introduce new FWI behavior; it reuses the existing AcousticFWI path and backend API.

## Usage

```bash
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device auto --prefer npu,cpu
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device npu:0
```

## Validation

Validated on 2026-05-26 in the `adfwi` conda environment.

- `python -m py_compile scripts/examples/minimal_acoustic_fwi_backend.py`: passed.
- `conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device cpu`: passed; loss `3.8032811744415085e-07`, `vp_grad_norm=1.8180083216634557e-08`, `vp_update_norm=1.8180636167526245`.
- `conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device npu:0`: passed; loss `3.8032811744415085e-07`, `vp_grad_norm=1.8180081440277718e-08`, `vp_update_norm=1.818063735961914`.

The CPU and NPU runs both returned finite nonzero gradients and model updates while writing no output files.
