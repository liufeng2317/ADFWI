# 91. Marmousi2 Preset Python Profiling Option

## Optimization Path

Add an optional profiling wrapper to the fixed Marmousi2 full-case preset path:

```text
fixed preset -> optional cProfile wrapper -> full-case output -> optional compare
```

This targets Python-side FWI bookkeeping diagnosis before changing FWI core
code. It does not profile NPU kernels directly; it is intended to separate
Python overhead from propagator/NPU runtime before deeper optimization work.

## Code Path

| Path | Change |
| --- | --- |
| `scripts/benchmark/run_marmousi2_full_case.py` | Add `--profile` and `--profile-output` to run presets under Python `cProfile`. |
| `tests/test_marmousi2_full_case_presets.py` | Add dry-run coverage for profile command construction. |
| `tests/full_cases/README.md` | Document profiling with preset and compare commands. |

## Usage

Dry-run a profiled preset:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --dry-run \
  --profile \
  --output-dir tests/full_cases/outputs/marmousi2_profiled \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,profiled
```

Run profiling and compare against a saved baseline:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --overwrite \
  --profile \
  --output-dir tests/full_cases/outputs/marmousi2_profiled \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,profiled
```

The default profile output is:

```text
<output-dir>/python_profile.prof
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
Ran 9 tests in 1.648s
OK
```

Dry-run profile and compare command check:

```bash
conda run -n adfwi python scripts/benchmark/run_marmousi2_full_case.py shot3 \
  --dry-run \
  --profile \
  --output-dir tests/full_cases/outputs/marmousi2_profiled \
  --compare-to tests/full_cases/outputs/marmousi2_npu_shot3_ckpt10_iter10 \
  --compare-labels baseline,profiled
```

The dry-run JSON includes:

- `base_command`: the unprofiled full-case command;
- `command`: the `python -m cProfile -o ...` wrapped command;
- `profile_output`: `tests/full_cases/outputs/marmousi2_profiled/python_profile.prof`;
- `compare_command`: the post-run baseline comparison command.

Saved-output comparison smoke:

```text
ok shot3 0.5028282545506961
```

## Numerical Precision

No FWI core code changed. This is a profiling workflow option only. The
underlying full-case command remains unchanged and can still be compared with
`--compare-to` after the profiled run completes.

## Next Direction

Run the profiled `shot3` preset once when runtime is available, then inspect
`python_profile.prof` to decide whether the next optimization should target:

1. Python-side FWI loop bookkeeping;
2. gradient processing; or
3. propagator/NPU operator runtime, which would require a deeper torch/NPU
   profiler pass rather than more `cProfile` work.
