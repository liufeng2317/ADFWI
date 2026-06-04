# Phase B Forward Parity Start

## 1. Overall Target

```text
Deepwave-style acoustic optimization
  |
  +-- Phase A: contract and baseline complete
  |
  +-- Phase B: isolated acoustic forward prototype
        |
        +-- current round: standardize parity entry and output location
        +-- next round: receiver-output parity gate
```

The long-term target is a custom acoustic operator with explicit forward,
backward, and storage contracts. This round stays at the Phase B boundary.

## 2. Current Focus

Focus:

- standardize benchmark/prototype output locations under `develope/`;
- use the existing `acoustic_custom_kernel_parity_probe.py` as the first Phase B
  parity entry instead of adding another probe;
- run a tiny parity smoke after the path cleanup.

Boundary:

- no production propagator changes;
- no new public `AcousticPropagator` option;
- no elastic work;
- no full 40-shot validation in this round.

## 3. Why This First

The repository already contains prototype scripts from earlier performance
work. Before implementing a deeper custom operator, these entry points must
write results into the current Deepwave branch records and be usable as the
standard parity gate.

## 4. Test Plan

Run a tiny synthetic parity smoke:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 1 \
  --nx 24 \
  --nz 18 \
  --nabc 6 \
  --nt 12 \
  --receivers 8
```

Required result:

- receiver output finite;
- raw `v.grad` finite;
- output and gradient difference recorded;
- result file written under `develope/`.

This is a parity smoke, not a speed acceptance test.

## 5. Result

Command executed:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_kernel_parity_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --shots 1 \
  --nx 24 \
  --nz 18 \
  --nabc 6 \
  --nt 12 \
  --receivers 8
```

Result file:

`develope/acoustic_custom_kernel_parity_probe_20260531.json`

Summary:

| Metric | Value |
| --- | ---: |
| output max abs diff | 0.0 |
| output max rel diff | 0.0 |
| loss abs diff | 0.0 |
| `v.grad` max abs diff | 5.169878828456423e-26 |
| `v.grad` max rel diff | 5.169878807280599e-14 |
| tiny-case forward speedup | 4.800662683111406x |
| tiny-case backward speedup | 1.8498587613787565x |
| tiny-case total speedup | 3.4624271296209828x |

Interpretation:

- The existing experimental acoustic forward prototype is usable as the first
  Phase B parity entry.
- The tiny-case speedup is only a direction signal. It is not a production
  performance claim.
- The next step should move from this synthetic parity smoke to a reduced
  Marmousi2 one-iteration parity test.

## 6. Next Boundary

Next round:

- run `acoustic_experimental_forward_iteration_parity.py` on the reduced
  Marmousi2 case;
- compare receiver outputs, loss, raw `vp.grad`, timing, and peak memory;
- keep production propagator unchanged.

Do not start full 40-shot validation until the reduced one-iteration parity
gate passes.
