# Marmousi2 Forward Case Check

## Purpose

The previous Marmousi2 case check verified that the real acoustic case can be loaded and reconstructed on CPU/NPU. This pass strengthens the optional `--case-run-forward` path so it validates one real single-shot forward calculation with a clear numerical contract.

The default `case-checks` suite remains read-only and lightweight. Forward execution is still opt-in because the real Marmousi2 geometry is significantly heavier than the minimal examples.

## Implementation

`marmousi2_acoustic_backend_check.py` now records tensor norms for model tensors, damping tensors, and optional forward pressure output. When `--run-forward` is enabled, it validates:

- pressure shape is `[1, nt, receiver_count]`;
- pressure values are finite;
- pressure norm is finite and nonzero;
- output device and dtype are included in the JSON summary.

`run_backend_smoke_suite.py` now compares `single_shot_forward.pressure.norm` across CPU and non-reference devices when `--case-run-forward` is enabled. The default tolerances are:

- `--case-forward-rtol 1e-4`;
- `--case-forward-atol 1e-6`.

A metric fails only when both absolute and relative tolerances are exceeded. This is intentional because the current Marmousi2 single-shot pressure norm is small.

## Commands

Read-only case check:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
```

Opt-in single-shot forward comparison:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic --case-run-forward --case-shot-index 0
```

## Validation Results

Validation commands passed:

```bash
python -m py_compile scripts/examples/marmousi2_acoustic_backend_check.py scripts/smoke/run_backend_smoke_suite.py
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-checks --devices cpu,npu:0 --case-checks marmousi2-acoustic --case-run-forward --case-shot-index 0
```

Observed single-shot forward result for shot `0`:

| Device | Pressure shape | Pressure norm | Seconds |
| --- | ---: | ---: | ---: |
| CPU | `[1, 3000, 200]` | `0.0019790849182754755` | `40.18367249146104` |
| NPU | `[1, 3000, 200]` | `0.0019793000537902117` | `8.125650119036436` |

CPU-vs-NPU forward norm comparison:

- absolute difference: `2.1513551473617554e-07`;
- relative difference: `0.00010870453953215872`;
- tolerance: `rtol=1e-4`, `atol=1e-6`;
- status: `ok`, because the absolute difference is below `1e-6`.

The runner summary reported `runs: 2`, `ok: 2`, `failed: 0`, `comparisons: 1`, `max_abs_diff: 2.1513551473617554e-07`, and `max_rel_diff: 0.00010870453953215872`.

## Notes

This remains a forward-path validation, not an inversion-quality benchmark. It does not write wavefields, figures, notebooks, or inversion outputs. The next step can use this foundation for a reduced Marmousi2 inversion smoke that checks the backward/gradient path on a small shot subset.
