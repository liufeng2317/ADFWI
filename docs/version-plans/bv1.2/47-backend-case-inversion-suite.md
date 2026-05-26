# Backend Case Inversion Suite

## Purpose

This optimization brings the reduced Marmousi2 acoustic inversion smoke into the
layered backend suite runner. The goal is to give future bv1.2 changes a
real-case backward/gradient regression check without touching notebooks or
writing generated example outputs.

The suite is intentionally opt-in because the CPU path is slower than the
lightweight public and tensor-level checks.

## Implementation

`run_backend_smoke_suite.py` now supports:

```text
--suites case-inversion
```

The first registered case inversion is:

```text
marmousi2-acoustic-reduced -> scripts/examples/marmousi2_acoustic_reduced_inversion.py
```

Default settings:

```text
shot_count = 1
nt_samples = 300
misfit = safe-squared-l2
lr = 1e12
dt_for_loss = 1.0
grad_mute_top = 12
model_file = init_model.npz
```

The runner records the exact child command, backend diagnostics, seed, case path,
subset size, model summary, and inversion metrics printed by the child script.

## Comparison Metrics

When CPU and an accelerator are both requested, CPU is used as the reference.
The suite compares:

```text
loss
vp_grad_norm
vp_update_norm
```

Default tolerances:

```text
case_inversion_rtol = 1e-4
case_inversion_atol = 1e-8
```

These tolerances are deliberately absolute-tolerance friendly because reduced
FWI gradients can be very small.

## Commands

CPU/NPU reduced Marmousi2 inversion check:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices cpu,npu:0
```

NPU-only fast check:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices npu:0
```

Custom reduced window:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices cpu,npu:0 --case-inversion-shot-count 1 --case-inversion-nt-samples 300
```

## Validation Result

Command:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites case-inversion --devices cpu,npu:0
```

Result: passed.

| Metric | CPU reference | NPU value | Absolute diff | Relative diff | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| `loss` | `4.9887580644281115e-06` | `4.9887585191754624e-06` | `4.547473508864641e-13` | `9.11544206019929e-08` | OK |
| `vp_grad_norm` | `4.905012621113716e-14` | `4.905012959926895e-14` | `3.3881317890172014e-21` | `3.3881317890172014e-13` | OK |
| `vp_update_norm` | `0.04906421899795532` | `0.04906421899795532` | `0.0` | `0.0` | OK |

Runtime on the local machine:

```text
CPU inversion seconds: 64.69841476716101
NPU inversion seconds: 4.9541123528033495
```

Suite summary:

```text
runs = 2
ok = 2
failed = 0
max_abs_diff = 4.547473508864641e-13
max_rel_diff = 9.11544206019929e-08
```

## Outcome

The backend smoke runner now covers the real-case forward/backward path:

1. `case-checks` validates read-only Marmousi2 data/model/survey loading and
   optional single-shot forward propagation;
2. `case-inversion` validates reduced Marmousi2 loss, backward gradient, and
   optimizer update on CPU/NPU.

This creates a practical regression gate for upcoming FWI loop, propagator,
backend, and transform refactors.
