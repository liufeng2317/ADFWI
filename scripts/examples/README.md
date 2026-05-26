# ADFWI Script Examples

This directory contains script-style examples for researchers who want to use ADFWI without editing notebook outputs. The examples are intentionally small, deterministic, and safe to run on CPU or NPU machines.

## Recommended First Commands

Run the acoustic example first. It is the smallest complete FWI path:

```bash
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device auto --prefer npu,cpu
```

Run the elastic example next if your machine supports the selected backend:

```bash
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device auto --prefer npu,cpu
```

Use explicit devices when comparing machines or debugging backend behavior:

```bash
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_acoustic_fwi_backend.py --device npu:0
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device cpu
conda run -n adfwi python scripts/examples/minimal_elastic_fwi_backend.py --device npu:0
```

## What These Examples Check

| Script | Physical path | Inversion step | Output behavior |
| --- | --- | --- | --- |
| `minimal_acoustic_fwi_backend.py` | Acoustic pressure modeling | One `AcousticFWI` iteration updating `vp` | JSON only |
| `minimal_elastic_fwi_backend.py` | Isotropic elastic modeling, pressure component | One `ElasticFWI` iteration updating `vp` | JSON only |
| `marmousi2_acoustic_backend_check.py` | Existing Marmousi2 acoustic case | Rebuild model, survey, observed data, and propagator; optional one-shot forward | JSON only |
| `marmousi2_acoustic_reduced_inversion.py` | Existing Marmousi2 acoustic case subset | One reduced `AcousticFWI` iteration for backward/gradient smoke | JSON only |

Both examples:

- configure the framework once with `ADFWI.set_backend(...)`;
- build tiny true and initial models in memory;
- synthesize observed data in memory;
- run one FWI iteration;
- verify finite positive loss, `vp` gradient norm, and model update norm;
- write no notebooks, figures, wavefields, or example data files.

## One-Line Backend Pattern

Use this pattern before creating models, propagators, regularization objects, or FWI drivers:

```python
import ADFWI

ADFWI.set_backend("npu:0", dtype="float32")
```

For automatic selection on CPU/NPU machines:

```python
ADFWI.set_backend(None, prefer="npu,cpu")
```

For CPU-only debugging:

```python
ADFWI.set_backend("cpu", dtype="float32")
```

## Validate Examples With The Smoke Suite

The layered smoke runner can execute both examples and compare CPU against NPU numerical metrics:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic,elastic
```

The runner reports a top-level `summary`. For the `examples` suite, check:

- `summary.by_suite[].runs`
- `summary.by_suite[].ok`
- `summary.by_suite[].failed`
- `summary.by_suite[].max_abs_diff`
- `summary.by_suite[].max_rel_diff`

The detailed `reports` block remains available when you need per-script loss, gradient norm, update norm, timing, dtype, and device information.

## Check The Existing Marmousi2 Case

The Marmousi2 check script is a bridge from minimal examples to the real notebook case. By default it is read-only: it loads the existing model and observed waveform files, rebuilds ADFWI objects, initializes the propagator, and prints a JSON summary. It does not rewrite notebook outputs or `examples/acoustic/.../data` files.

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device cpu
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device npu:0
```

Use `--run-forward --shot-index 0` only when you intentionally want to run one selected shot through the acoustic propagator. The default mode is intended for fast case integrity checks before heavier benchmark or inversion runs.

Run the reduced inversion smoke when you want to verify the real-case backward and gradient path without running the full notebook inversion:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device cpu --shot-count 1 --nt-samples 300
```

The reduced inversion script uses the package-level `Misfit_waveform_SquaredL2(reduction="mean")` through the `safe-squared-l2` option by default. The legacy waveform L2 misfit is available with `--misfit legacy-l2` for diagnostics, but it can produce NaN gradients on this reduced window because its square-root form is singular at zero residual.

The same check can be launched through the layered smoke runner:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic --case-run-forward --case-shot-index 0
```

## Extending Toward A Real Case

When adapting a minimal script to a real example such as Marmousi2, keep the same outer structure and replace the synthetic pieces gradually:

| Minimal function or block | Replace with real-case logic |
| --- | --- |
| `model_arrays(...)` | Load or construct real `vp`, `vs`, and `rho` arrays |
| `build_survey(...)` | Load real source/receiver geometry and wavelets |
| observed-data synthesis block | Load observed field data or synthetic benchmark observations |
| `Misfit_waveform_L2` | Choose the target misfit from `ADFWI.fwi.misfit` |
| `GradProcessor(...)` | Configure illumination, normalization, and clipping policies |
| optimizer and scheduler | Use the optimization strategy for the target inversion |

Keep the backend call at the top of the script. That gives CPU/NPU/CUDA switching one stable entry point and avoids scattering device strings across models, propagators, and FWI objects.

## When To Use Notebooks

Use notebooks for exploration, plotting, and interpretation. Use scripts in this directory for reproducible smoke checks, backend migration, and CPU/NPU comparisons. Script examples should avoid persistent outputs unless the filename and output directory are explicit user-facing choices.
