# 90. Marmousi2 Preset Post-Run Compare

## Optimization Path

`scripts/benchmark/run_marmousi2_full_case.py` now supports a post-run compare
step:

```text
fixed preset -> full-case output directory -> compare_full_case_outputs.py
```

This keeps the agreed validation loop short for future code changes: run a
fixed `shot3` or `shot5` baseline, then immediately compare the new output with
a saved baseline using the same command.

## Code Path

| Path | Change |
| --- | --- |
| `scripts/benchmark/run_marmousi2_full_case.py` | Add `--compare-to`, `--compare-labels`, and optional loss-drift gate flags. |
| `tests/test_marmousi2_full_case_presets.py` | Add dry-run coverage for generated compare commands. |
| `tests/full_cases/README.md` | Document post-run compare usage. |

## Usage

Dry-run the full run plus compare plan:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --dry-run \
  --output-dir tests/full_cases/outputs/marmousi2_candidate \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,candidate
```

Run and compare:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --overwrite \
  --output-dir tests/full_cases/outputs/marmousi2_candidate \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,candidate
```

Optional regression gate:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --overwrite \
  --output-dir tests/full_cases/outputs/marmousi2_candidate \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,candidate \
  --fail-on-loss-drift
```

## Test Comparison

Lightweight tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_marmousi2_full_case_presets.py \
  tests/test_full_case_output_compare.py
```

Result:

```text
Ran 8 tests in 1.286s
OK
```

Dry-run comparison command check:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --dry-run \
  --output-dir tests/full_cases/outputs/marmousi2_candidate \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,candidate
```

The dry-run JSON includes both:

- the full-case inversion command; and
- the generated `compare_full_case_outputs.py` command.

Saved-output comparison smoke:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
  --labels shot3,shot5
```

Key output:

```text
ok shot3 0.5028282545506961
```

## Numerical Precision

No FWI core code changed. This is a benchmark workflow optimization only, so no
new FWI numerical precision comparison was required. The comparison path itself
was validated against saved 3-shot and 5-shot outputs.

## Next Direction

With fixed run and compare commands in place, the next useful optimization can
move back to code behavior. Candidate paths:

1. profile the fixed `shot3` baseline to identify whether the next bottleneck is
   propagator runtime, gradient processing, or Python-side FWI bookkeeping; or
2. validate the existing opt-in `TorchGradProcessor` on a full-case preset before
   considering a broader torch-native migration.
