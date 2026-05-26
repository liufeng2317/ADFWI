# Backend Case Check Suite

## Purpose

The Marmousi2 acoustic backend check is useful on its own, but it should also be reachable through the same layered smoke runner used for public API, misfit, forward, minimal examples, and mini-inversion checks. This pass adds a `case-checks` suite to `scripts/smoke/run_backend_smoke_suite.py`.

## Implementation

The runner now supports:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
```

The suite calls `scripts/examples/marmousi2_acoustic_backend_check.py` once per requested device. It remains read-only by default and writes no notebook outputs, figures, wavefields, inversion files, or data files.

Optional stronger checks can be requested with:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices npu:0 --case-run-forward --case-shot-index 0
```

That optional mode runs one selected shot through the real Marmousi2 propagator and is intentionally not enabled by default.

## Validation

Validation commands for this pass:

```bash
python -m py_compile scripts/smoke/run_backend_smoke_suite.py scripts/examples/marmousi2_acoustic_backend_check.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
```

The expected result is a JSON report with `status: ok`, `summary.runs: 2`, and both CPU/NPU child runs marked `ok`.

Local validation reported `status: ok`, `summary.runs: 2`, `summary.ok: 2`, `summary.failed: 0`, and `summary.unavailable: 0`. The NPU child run confirmed `vp`, `rho`, propagator device, and damping tensor on `npu:0`.

## Scientific Contract

The case-check suite validates real-case object construction rather than inversion quality. The current contract is:

- Marmousi2 model and observed waveform files are present;
- `AcousticModel`, `Survey`, `SeismicData`, and `AcousticPropagator` can be reconstructed;
- `vp`, `rho`, observed `p/u/w`, and damping tensors are finite;
- observed pressure has nonzero norm;
- NPU runs place model tensors and damping on `npu:0`;
- no persistent outputs are written.
