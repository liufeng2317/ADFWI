# Reduced Marmousi2 One-Iteration Parity

## 1. Overall Target

```text
Deepwave-style acoustic optimization
  |
  +-- Phase B: isolated acoustic forward prototype
        |
        +-- tiny synthetic parity: passed
        +-- reduced Marmousi2 one-iteration parity: current result
```

This round checks whether the existing experimental acoustic forward prototype
can be embedded in the reduced Marmousi2 FWI call chain without changing the
production propagator.

## 2. Current Focus

Focus:

- compare production and experimental acoustic forward paths in one reduced
  Marmousi2 FWI-style iteration;
- record receiver output parity, loss parity, raw `vp.grad` behavior, timing,
  and peak memory.

Boundary:

- no production propagator changes;
- no full 40-shot validation;
- no public API change;
- no elastic work.

## 3. Tests

### Observed-Pressure Loss

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32
```

Result file:

`develope/acoustic_experimental_forward_iteration_parity_20260531.json`

| Metric | Result |
| --- | ---: |
| loss abs diff | 0.0 |
| output max abs diff | 0.0 |
| output max rel diff | 0.0 |
| reference raw grad finite | false |
| candidate raw grad finite | false |
| total speedup | 1.557356653990944x |
| peak memory ratio | 1.8320097782927554x |

### Observed-Pressure Loss With Waveform Normalization

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --waveform-normalize \
  --result-json docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_experimental_forward_iteration_parity_observed_normalized_20260604.json
```

Result file:

`develope/acoustic_experimental_forward_iteration_parity_observed_normalized_20260604.json`

| Metric | Result |
| --- | ---: |
| loss abs diff | 0.0 |
| output max abs diff | 0.0 |
| output max rel diff | 0.0 |
| reference raw grad finite | false |
| candidate raw grad finite | false |
| total speedup | 1.5896925574613772x |
| peak memory ratio | 1.8309061146155257x |

### Synthetic-Energy Loss

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --loss-mode synthetic-energy \
  --result-json docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_experimental_forward_iteration_parity_synthetic_energy_20260604.json
```

Result file:

`develope/acoustic_experimental_forward_iteration_parity_synthetic_energy_20260604.json`

| Metric | Result |
| --- | ---: |
| loss abs diff | 0.0 |
| output max abs diff | 0.0 |
| output max rel diff | 0.0 |
| raw `vp.grad` max abs diff | 3.552713678800501e-15 |
| raw `vp.grad` max rel diff | 1.859006033555488e-06 |
| reference raw grad finite | true |
| candidate raw grad finite | true |
| total speedup | 1.5422803503372968x |
| peak memory ratio | 2.443922175284108x |

## 4. Interpretation

- Receiver output and loss are exactly matched in all reduced Marmousi2 tests.
- The synthetic-energy loss confirms that the experimental path can produce a
  finite and tolerance-bounded raw `vp.grad` in the reduced Marmousi2 geometry.
- The observed-pressure loss is not a valid raw-gradient parity gate in this
  script configuration because the production reference raw gradient is already
  non-finite.
- The speed numbers remain directional. The experimental path is faster in this
  one-iteration reduced case, but peak memory is higher.

## 5. Decision

Keep Phase B active. The next optimization should not jump to full validation.
First, the observed-pressure non-finite raw-gradient gate needs to be made
diagnostic:

- report non-finite count and first non-finite index for production and
  candidate gradients;
- decide whether observed-pressure parity should compare processed gradients
  instead of raw gradients;
- keep synthetic-energy as the current raw-gradient parity gate.

## 6. Observed-Pressure Gradient Diagnostic

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --waveform-normalize \
  --result-json docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_experimental_forward_iteration_parity_observed_gradient_diagnostic_20260604.json
```

Result file:

`develope/acoustic_experimental_forward_iteration_parity_observed_gradient_diagnostic_20260604.json`

| Metric | Production | Experimental |
| --- | ---: | ---: |
| finite entries | 64 | 64 |
| NaN entries | 1984 | 1984 |
| +Inf entries | 0 | 0 |
| -Inf entries | 0 | 0 |
| first non-finite index | `[1, 0]` | `[1, 0]` |
| finite min/max | 0.0 / 0.0 | 0.0 / 0.0 |

Interpretation:

- The observed-pressure raw-gradient non-finite pattern is identical in
  production and experimental paths.
- This confirms that the observed-pressure raw-gradient failure is not a
  candidate-specific regression.
- For Phase B, raw-gradient parity should continue to use the synthetic-energy
  gate until the observed-pressure gradient normalization/zero-amplitude
  behavior is handled separately.

Next boundary:

- use reduced Marmousi2 synthetic-energy as the raw-gradient parity gate;
- use observed-pressure only for receiver output and loss parity for now;
- next optimization can move to a short `3 shot x 3 iteration` loop only if it
  records processed gradient finite checks instead of raw observed-pressure
  gradient parity.
