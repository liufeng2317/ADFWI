# 88. Full-Case Output Compare Tool

## Purpose

Make the fixed Marmousi2 full-case baselines easier to reuse for future
optimization work. Previous comparisons required manual parsing of
`summary.json`; this record adds a script that compares saved full-case outputs
directly and reports timing, loss, gradient, and loss-history drift in one JSON
report.

## Code Path

| Path | Change |
| --- | --- |
| `scripts/benchmark/compare_full_case_outputs.py` | New CLI for comparing full-case output directories or `summary.json` files. |
| `tests/test_full_case_output_compare.py` | Unit tests for summary loading, run comparison, CLI JSON output, and loss-drift failure mode. |
| `scripts/examples/marmousi2_acoustic_reduced_inversion.py` | Record `checkpoint_segments` in future full-case `summary.json` files. |
| `tests/full_cases/README.md` | Document the comparison command for saved full-case outputs. |

## Usage

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
  --labels shot3,shot5
```

Optional loss-drift gate:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  baseline_dir candidate_dir \
  --labels baseline,candidate \
  --fail-on-loss-drift \
  --loss-abs-tol 1e-5 \
  --loss-rel-tol 1e-8
```

## Validation

Lightweight tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_full_case_output_compare.py \
  tests/test_marmousi2_reduced_inversion.py \
  tests/full_cases/test_marmousi2_acoustic_full_flow.py
```

Result:

```text
Ran 9 tests in 11.807s
OK (skipped=1)
```

Saved-output comparison smoke:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
  --labels shot3,shot5
```

Key output:

| Metric | Value |
| --- | ---: |
| status | ok |
| reference | shot3 |
| shot3 seconds_per_iteration | 31.358286083862186 |
| shot5 seconds_per_iteration | 31.861114338412882 |
| seconds_per_iteration_abs_diff | 0.5028282545506961 |
| loss_history_length_match | true |

## Numerical Impact

No FWI core numerical path changed. The only runtime script metadata change is
that future full-case summaries now include `subset.checkpoint_segments`.
Existing saved outputs remain readable; older summaries report that field as
`null`.

## Next Step

Use this compare tool after each future FWI code optimization against the fixed
3-shot or 5-shot full-case baselines. The next useful optimization is to expose
the fixed baseline configurations through a small reusable command wrapper or
configuration preset so repeated runs and comparisons require fewer environment
variables.
