# 93. Compatibility Cleanup Audit

## Purpose

Audit the compatibility surfaces kept during the bv1.2 framework refactor and
plan a staged cleanup. The goal is to start removing or narrowing compatibility
layers without accidentally changing legacy numerical behavior.

This record is a planning and statistics pass only. No FWI core code changed.

## Audit Commands

```bash
git status --short --branch
find ADFWI -type f \( -name '*normalization*' -o -name '*multiScale*' -o -name '*multiscale*' -o -name '*compat*' \) | sort
rg -n "ADFWI\.fwi\.normalization|ADFWI\.fwi\.multiScaleProcessing|from ADFWI\.fwi\.multiScaleProcessing|from ADFWI\.fwi import multiScaleProcessing" ADFWI scripts tests examples docs --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
rg -n "LegacyLowPassFilter|legacy-lowpass|lowpass-mode|ADFWI\.fwi\.multiscale|from ADFWI\.fwi\.multiScaleProcessing|lpass\(" ADFWI scripts tests examples docs --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
rg -n "legacy-l2|Misfit_waveform_L2|legacy L2" ADFWI scripts tests examples docs --glob '!tests/full_cases/outputs/**' --glob '!**/__pycache__/**'
```

## Summary Statistics

| Category | Count | Meaning |
| --- | ---: | --- |
| tracked `ADFWI/fwi/**/__pycache__/*` files | 0 | Runtime cache files are not tracked; ignore for cleanup commits. |
| old shim/import references | 7 files | Mostly compatibility tests plus the shim files themselves. |
| canonical normalization references | 8 files | Current code already uses `TraceNormalize` or transform-layer normalization. |
| legacy low-pass references | 14 files | Still an intentional legacy numerical path and test/smoke target. |
| gradient processor references | many | Broad public API and example surface; do not remove as a shim cleanup. |
| legacy L2 references | 19 files | Intentional diagnostic/full-case path; numerical behavior must be preserved. |
| tracked `ADFWI/fwi/*/backup` files | 0 | Backup directories are present locally but not tracked by git. |

## Compatibility Surfaces

| Surface | Current owner | Compatibility path | Current status | Cleanup risk |
| --- | --- | --- | --- | --- |
| Waveform normalization helper | `ADFWI.fwi.transforms.amplitude.normalize_waveform` | `ADFWI.fwi.normalization.normalize_waveform` | Internal production code no longer depends on the old path. | Low |
| Legacy multiscale low-pass implementation | `ADFWI.fwi.multiscale.legacy_lowpass` through `ADFWI.fwi.multiscale` | `ADFWI.fwi.multiScaleProcessing` | Internal production code uses canonical imports; tests keep old import coverage. | Low for imports, high for numerical replacement |
| `ADFWI.fwi.transforms.waveform` exports | Split transform modules | `ADFWI.fwi.transforms.waveform` | Thin compatibility export for older transform imports. | Low |
| `ADFWI.fwi.data.normalize_waveform` | Transform amplitude helper | Data package re-export | Kept for historical data-layer imports. | Low |
| Legacy low-pass behavior | `LegacyLowPassFilter` and `lpass` | `cutoff_freq` driver behavior | Deliberate numerical compatibility path. | High |
| Legacy L2 misfit | `Misfit_waveform_L2` | `--misfit legacy-l2` and full-case baselines | Deliberate diagnostic/full-case path. | High |
| `GradProcessor` | `ADFWI.propagator.gradient_process.GradProcessor` | Default gradient processor in examples/FWI docs | Public API and legacy numerical behavior. | High |
| `TorchGradProcessor` | `ADFWI.propagator.gradient_process.TorchGradProcessor` | Opt-in candidate path | Not compatibility debt; candidate migration path. | Medium |

## Immediate Findings

- The thin files `ADFWI/fwi/normalization.py` and
  `ADFWI/fwi/multiScaleProcessing.py` are genuine compatibility shims.
- The earlier internal import cleanup succeeded: remaining old shim references
  are compatibility tests, shim files, or historical documentation.
- `multiScaleProcessing.py` should not be deleted yet unless the public API
  break is explicitly accepted.
- Legacy low-pass, legacy L2, and `GradProcessor` are not just shim debt. They
  encode numerical behavior used by tests, examples, and current full-case
  baselines.
- Local `backup/` and `__pycache__` directories are not tracked by git, so they
  are outside this staged compatibility cleanup.

## Cleanup Plan

### Phase 1: Low-Risk Shim Deprecation

Add explicit deprecation warnings to thin compatibility shims while preserving
their exports:

- `ADFWI.fwi.normalization`
- `ADFWI.fwi.multiScaleProcessing`
- optionally `ADFWI.fwi.transforms.waveform`

Validation:

```bash
conda run -n adfwi python -m unittest \
  tests/test_data_transforms.py \
  tests/test_multiscale_compat.py \
  tests/test_lowpass_transform_comparison.py
```

Numerical precision requirement: no full-case rerun required if only warnings
and import surfaces change.

### Phase 2: Internal and Example Import Canonicalization

Keep public shims, but make examples and new docs prefer canonical imports:

- normalization: `ADFWI.fwi.transforms` or
  `ADFWI.fwi.transforms.amplitude`
- multiscale: `ADFWI.fwi.multiscale`
- waveform transforms: `ADFWI.fwi.transforms`

Validation:

```bash
conda run -n adfwi python -m unittest tests/test_data_transforms.py tests/test_multiscale_compat.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites examples --devices cpu,npu:0 --example-problems acoustic --example-gradient-processors legacy,torch
```

Numerical precision requirement: smoke comparison is enough unless a numerical
method changes.

### Phase 3: Legacy Numerical Path Guardrails

Do not delete or silently replace these paths:

- `LegacyLowPassFilter`
- `Misfit_waveform_L2` and `--misfit legacy-l2`
- `GradProcessor`

Instead, make legacy status explicit in docs and CLI help, and keep candidate
paths opt-in:

- `LowPassFilter`
- `safe-squared-l2`
- `TorchGradProcessor`

Validation for any numerical-path change:

```bash
conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py tests/test_lowpass_transform_comparison.py
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --overwrite \
  --output-dir tests/full_cases/outputs/marmousi2_candidate \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,candidate
```

### Phase 4: Future Removal Gate

Only consider removing shim modules in a later version after:

1. one release cycle with deprecation warnings;
2. no internal imports use the old path;
3. user-facing docs point to canonical imports;
4. migration notes explicitly list replacements.

## Recommended Next Commit

Start with Phase 1: add warnings to `ADFWI.fwi.normalization` and
`ADFWI.fwi.multiScaleProcessing`, then run the focused import/transform tests.
This is the lowest-risk compatibility cleanup because it does not alter FWI
numerical behavior or public exports.
