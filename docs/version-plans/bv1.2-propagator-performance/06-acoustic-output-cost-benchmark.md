# Acoustic Output-Cost Benchmark

Date: 2026-05-31

## Boundary

```text
Goal:
  Add a benchmark that isolates acoustic receiver sampling and forward-
  wavefield accumulation cost.

Scope:
  Benchmark script and documentation only.

Validation:
  py_compile, CPU smoke run, NPU representative output-cost run, and NPU
  reduced-shape output-cost run.

Stop:
  Stop after measuring output-side cost. Do not modify the production acoustic
  kernel, finite-difference equations, output semantics, or FWI precision.
```

## Benchmark Added

File:

```text
scripts/benchmark/acoustic_output_cost.py
```

The benchmark repeats the output-side tensor expressions used in
`ADFWI/propagator/acoustic_kernels.py`:

```text
receiver sampling:
  p[:, rcv_z, rcv_x]
  u[:, rcv_z, rcv_x]
  w[:, rcv_z, rcv_x]

forward-wavefield accumulation:
  torch.sum(p * p, dim=0)[nabc:nabc+nz, nabc:nabc+nx].detach()
  torch.sum(u * u, dim=0)[nabc:nabc+nz, nabc:nabc+nx].detach()
  torch.sum(w * w, dim=0)[nabc:nabc+nz, nabc:nabc+nx].detach()
```

This is a micro-benchmark, not an alternative propagator. It is only intended
to decide whether output recording/accumulation is large enough to justify a
future kernel-level optimization.

## Modes

| Mode | Receiver output | Forward wavefield |
| --- | --- | --- |
| `empty` | none | none |
| `receiver_p` | `p` only | none |
| `receiver_puw` | `p/u/w` | none |
| `wavefield_p` | none | `p` only |
| `wavefield_puw` | none | `p/u/w` |
| `receiver_puw_wavefield_puw` | `p/u/w` | `p/u/w` |

When `--requires-grad` is enabled, backward timing is reported for receiver
modes because receiver pressure can feed a scalar loss. Wavefield accumulation
uses `.detach()` in the production kernel, so pure wavefield modes have no
backward path.

## Static And CPU Smoke

Compile:

```bash
conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_output_cost.py
```

CPU smoke:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_output_cost.py \
  --device cpu \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 1 \
  --receivers 8 \
  --nx 24 \
  --nz 20 \
  --nabc 4 \
  --nt 10 \
  --requires-grad
```

Result:

```text
status: ok
all output tensors finite
receiver modes produce nonzero receiver records
wavefield modes produce nonzero forward-wavefield summaries
receiver backward produces finite p gradients
```

## NPU Representative Shape

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_output_cost.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --shots 1 \
  --receivers 200 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --nt 800 \
  --requires-grad \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_output_cost_20260531.json
```

Summary:

| Mode | Forward mean (s) | Backward mean (s) | Total mean (s) |
| --- | ---: | ---: | ---: |
| `empty` | 0.0003 | n/a | 0.0003 |
| `receiver_p` | 0.1031 | 0.2543 | 0.3574 |
| `receiver_puw` | 0.3065 | 0.2537 | 0.5602 |
| `wavefield_p` | 0.0870 | n/a | 0.0870 |
| `wavefield_puw` | 0.2628 | n/a | 0.2628 |
| `receiver_puw_wavefield_puw` | 0.6331 | 0.2536 | 0.8867 |

Interpretation:

```text
receiver p/u/w sampling forward cost: about 0.306 s
wavefield p/u/w accumulation forward cost: about 0.263 s
combined output-side forward cost: about 0.633 s
```

Compared with the current single-shot differentiable acoustic benchmark total
time of about `6.55 s`, output-side operations are visible but are not the
dominant cost.

## NPU Reduced-Shape Probe

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_output_cost.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 3 \
  --receivers 200 \
  --nx 170 \
  --nz 70 \
  --nabc 50 \
  --nt 3000 \
  --requires-grad \
  --modes receiver_puw,wavefield_puw,receiver_puw_wavefield_puw \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_output_cost_reduced_shape_20260531.json
```

Summary:

| Mode | Forward (s) | Backward (s) | Total (s) |
| --- | ---: | ---: | ---: |
| `receiver_puw` | 1.3916 | 1.2586 | 2.6502 |
| `wavefield_puw` | 1.1844 | n/a | 1.1844 |
| `receiver_puw_wavefield_puw` | 2.1314 | 0.7452 | 2.8765 |

Interpretation:

```text
For reduced validation-like dimensions, output-side work is large enough to
measure clearly. However, the benchmark is synthetic and does not include the
finite-difference update cost, so it should guide profiling rather than directly
justify changing production output semantics.
```

## Precision And Contract

No production code was changed in this round, so acoustic numerical precision is
not affected. The benchmark checks tensor shapes, dtype/device, finiteness, and
receiver-gradient finiteness where backward is meaningful.

## Remaining Risk

- This is a micro-benchmark with static random wavefields, not a full
  propagation benchmark.
- Backward timings for synthetic receiver-only graphs are useful for rough
  graph-cost visibility, but they are not identical to a real FWI backward pass.
- Full-record `40 shots, nt=3000` output-only timing was not run because it is
  expensive and the current goal was bottleneck direction, not final speedup.

## Next Direction

The next production optimization should not remove default outputs, because
that would change public behavior. A safe next step is to prototype an opt-in
recording policy in a benchmark branch first, for example pressure-only receiver
recording or optional forward-wavefield accumulation, and compare:

- default mode: exact output/gradient parity;
- opt-in mode: explicit output-key contract and FWI loss compatibility;
- real reduced validation timing.
