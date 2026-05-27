# 89. Marmousi2 Full-Case Preset Runner

## Purpose

Make the fixed Marmousi2 full-case baselines runnable without copying long
environment-variable commands. The new preset runner provides stable `shot3`
and `shot5` entry points that match the current bv1.2 benchmark records.

## Code Path

| Path | Change |
| --- | --- |
| `scripts/benchmark/run_marmousi2_full_case.py` | New preset CLI for running fixed Marmousi2 full-case baselines. |
| `tests/test_marmousi2_full_case_presets.py` | Unit tests for preset values, generated commands, dry-run JSON, and preset listing. |
| `tests/full_cases/README.md` | Document the new preset commands. |

## Presets

| Preset | Purpose | Key settings |
| --- | --- | --- |
| `shot3` | Fastest current full-case per-iteration NPU baseline | 3 shots, 10 iterations, 3000 samples, `checkpoint_segments=10` |
| `shot5` | Current full-case throughput stress baseline | 5 shots, 10 iterations, 3000 samples, `checkpoint_segments=10` |

Both presets use:

- device default: `npu:0`
- observed source: synthetic true model
- optimizer: Adam
- lr: `10`
- scheduler: `StepLR(step_size=200, gamma=0.75)`
- misfit: legacy L2
- waveform normalization: enabled
- auto rho update: enabled

## Usage

Dry-run the command without running the full case:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --dry-run
```

Run the fixed baselines:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 --overwrite
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot5 --overwrite
```

Compare saved outputs:

```bash
conda run -n adfwi python scripts/benchmark/compare_full_case_outputs.py \
  tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  tests/full_cases/outputs/marmousi2_npu_shot5_ckpt10_iter10 \
  --labels shot3,shot5
```

## Validation

Lightweight tests:

```bash
conda run -n adfwi python -m unittest \
  tests/test_marmousi2_full_case_presets.py \
  tests/test_full_case_output_compare.py
```

Result:

```text
Ran 7 tests in 1.092s
OK
```

Dry-run smoke:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py \
  shot3 \
  --dry-run \
  --output-dir tests/full_cases/outputs/marmousi2_dry_run
```

Key dry-run output:

| Field | Value |
| --- | --- |
| status | ok |
| preset | shot3 |
| shot_count | 3 |
| iterations | 10 |
| checkpoint_segments | 10 |
| output_dir | `tests/full_cases/outputs/marmousi2_dry_run` |

Saved-output comparison smoke:

```text
ok shot3 2 1
```

## Numerical Impact

No FWI numerical path changed. This is a benchmark workflow optimization only.
The full-case command produced by the `shot3` preset matches the current fixed
3-shot baseline settings.

## Next Step

Use the preset runner for future full-case validation after code changes. The
next code-level optimization should target a concrete bottleneck inside the
fixed baseline, then rerun `shot3` or `shot5` and compare outputs with
`compare_full_case_outputs.py`.
