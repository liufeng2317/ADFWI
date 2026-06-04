# Processed Gradient Gate

## 1. Overall Target

```text
Phase B reduced Marmousi2 parity
  |
  +-- receiver output parity
  +-- loss parity
  +-- raw gradient parity for synthetic-energy loss
  +-- processed gradient health for observed-pressure loss
```

The observed-pressure reduced parity showed identical non-finite raw gradients
in production and experimental paths. This round defines a processed-gradient
gate so the next short-loop tests can report useful gradient behavior without
misclassifying a production-side raw-gradient issue as an experimental
regression.

## 2. Gate Definition

For each production/candidate run, record:

| Field | Meaning |
| --- | --- |
| `raw_grad_health` | finite/NaN/Inf counts before gradient processing |
| `processed_grad_health` | finite/NaN/Inf counts after the configured gradient processor |
| `processed_grad_max_abs_diff` | numerical diff only meaningful when both processed gradients are finite |
| `processed_grad_max_rel_diff` | numerical diff only meaningful when both processed gradients are finite |

Pass/fail interpretation:

| Case | Decision |
| --- | --- |
| both processed gradients finite and diff is tolerance-bounded | pass |
| both processed gradients non-finite with identical pattern | diagnostic pass, not numerical pass |
| candidate non-finite while production finite | fail |
| production non-finite while candidate finite | production gate issue; record separately |

## 3. Current Implementation Boundary

- update only `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`;
- call the existing `fwi.process_gradient` path;
- pass `forw=None` because this parity script uses
  `grad_forw_illumination=False`;
- keep production propagator unchanged.

## 4. Test Plan

Run observed-pressure normalized reduced parity:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_forward_iteration_parity.py \
  --device npu:0 \
  --dtype float32 \
  --waveform-normalize \
  --result-json docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_experimental_forward_iteration_processed_gradient_20260604.json
```

Expected output:

- raw and processed gradient health for production and experimental paths;
- receiver output/loss parity unchanged;
- no production code changes.

## 5. Results

### Observed-Pressure With Waveform Normalization

Result file:

`develope/acoustic_experimental_forward_iteration_processed_gradient_20260604.json`

| Metric | Production | Experimental |
| --- | ---: | ---: |
| raw finite entries | 64 | 64 |
| raw NaN entries | 1984 | 1984 |
| processed finite entries | 0 | 0 |
| processed NaN entries | 2048 | 2048 |
| processed first non-finite index | `[0, 0]` | `[0, 0]` |

Output/loss parity remains exact:

| Metric | Value |
| --- | ---: |
| loss abs diff | 0.0 |
| output max abs diff | 0.0 |
| output max rel diff | 0.0 |
| total speedup | 1.5602538343085x |
| peak memory ratio | 1.831026064377743x |

Interpretation:

- The processed-gradient non-finite pattern is identical in production and
  experimental paths.
- Observed-pressure processed gradients cannot be used as a finite-gradient
  pass/fail gate in this reduced one-iteration configuration.

### Synthetic-Energy Loss

Result file:

`develope/acoustic_experimental_forward_iteration_processed_gradient_synthetic_energy_20260604.json`

| Metric | Value |
| --- | ---: |
| loss abs diff | 0.0 |
| output max abs diff | 0.0 |
| raw `vp.grad` max abs diff | 3.552713678800501e-15 |
| raw `vp.grad` max rel diff | 1.859006033555488e-06 |
| processed grad max abs diff | 0.000732421875 |
| processed grad max rel diff | 7.871763955336064e-06 |
| production processed grad finite | true |
| experimental processed grad finite | true |
| total speedup | 1.5609587608214228x |
| peak memory ratio | 2.443557141794929x |

Interpretation:

- The processed-gradient gate implementation works when the upstream loss
  produces finite raw gradients.
- Synthetic-energy remains the current finite raw/processed gradient parity
  gate for Phase B.
- Observed-pressure remains useful for receiver output and loss parity, but not
  for finite raw/processed gradient parity until the observed-loss gradient
  behavior is handled separately.

## 6. Decision

Do not proceed to full validation yet. The next practical optimization test can
be a short `3 shot x 3 iteration` loop only if it reports:

- observed-pressure loss trajectory and receiver/loss parity where applicable;
- synthetic-energy raw/processed gradient parity for the experimental operator;
- processed gradient health, not just raw gradient health;
- peak memory ratio.
