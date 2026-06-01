# Acoustic Observed-Scale Local Recurrence Probe

Date: 2026-05-31

## Purpose

The previous direct replay localized the mismatch to the experimental custom
backward under observed-pressure receiver upstream. This round keeps the test
at formula level and avoids another wrapper scan.

The question is:

```text
Can a small recurrence reproduce the mismatch when driven by receiver-pressure
linear upstream at the observed-loss scale?
```

## Script Change

`scripts/benchmark/acoustic_custom_multistep_update_probe.py` now supports:

- `--loss-kind receiver-random-linear`
- `--loss-components`
- `--upstream-seed`
- `--upstream-scale`

This keeps the original energy-loss probe available while adding a direct
receiver-upstream test path.

The script also records zero gradients for inputs that are intentionally unused
by a minimal loss, instead of treating `None` gradients as a script failure.

## Results

| Case | Device / dtype | Steps | Loss | Grad max abs diff | Grad max rel diff | Output diff |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| Regression energy loss | `npu:0` / `float32` | 20 | energy | `1.3346834748517722e-10` | `0.018632521852850914` | `0.0` |
| Observed-scale `rcv_p` upstream | `npu:0` / `float32` | 300 | linear | `5.109375` | `2.917783260345459` | `0.0` |
| Observed-scale `rcv_p` upstream | `cpu` / `float64` | 300 | linear | `5.5730345067313465` | `0.32208154173548736` | `0.0` |
| Observed-scale `rcv_p` upstream | `cpu` / `float32` | 300 | linear | `4.48828125` | `3.420240879058838` | `0.0` |
| Observed-scale `rcv_p` upstream | `cpu` / `float64` | 1 | linear | `0.0` | `0.0` | `0.0` |
| Observed-scale `rcv_p` upstream | `npu:0` / `float32` | 1 | linear | `0.0` | `0.0` | `0.0` |
| Observed-scale `rcv_p` upstream | `cpu` / `float64` | 2 | linear | `1.3612220087472204e-05` | `0.003620126831674235` | `0.0` |

## Decision

The mismatch is not caused by:

- production/FWI wrapper logic;
- receiver output values;
- the loss wrapper;
- single-step receiver-pressure backward;
- NPU-only numerical behavior.

The smallest failing case found here is:

```text
2-step recurrence + receiver-pressure linear upstream
```

Since `steps=1` passes exactly and `steps=2` fails in CPU `float64`, the
remaining issue is a formula-level mismatch in the multi-step backward
recurrence. The likely path is the adjoint contribution that passes from a
receiver pressure gradient through:

```text
p(t+1) -> u(t+1)/w(t+1) -> p(t+2)
```

## Next Direction

Do not continue wrapper/output/loss scans.

Next task:

```text
Build a 2-step local adjoint inspector that compares reference autograd and
custom backward gradients after each timestep, focusing on the p_new
contributions used by u and w updates.
```

Only after the 2-step local mismatch is fixed should the custom recurrence be
tested again on the full observed-upstream direct replay.
