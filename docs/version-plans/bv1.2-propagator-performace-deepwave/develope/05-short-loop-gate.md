# Short-Loop Gate

## 1. Overall Target

```text
Deepwave-style acoustic optimization
  |
  +-- Phase B parity gates complete
  |
  +-- short-loop gate: current result
        |
        +-- production and experimental both become non-finite
        +-- do not add more test-standard work
```

This round limits the second gate to one concrete question:

> Does the experimental acoustic forward path remain useful across a short
> `3 shot x 3 iteration` FWI-style loop?

## 2. Current Focus

Focus:

- run production vs experimental short loop;
- keep the case reduced: `3 shots`, `3 iterations`, `checkpoint_segments=10`;
- record loss trajectory, speed, and peak memory.

Boundary:

- no production propagator changes;
- no full 40-shot validation;
- no new test-gate category;
- no further expansion of testing standards.

## 3. Implementation Update

`scripts/benchmark/acoustic_experimental_fwi_loop_compare.py` was aligned with
the current gate:

- default candidate changed to `experimental`;
- default loop changed to `3` iterations;
- reduced geometry changed to `64 x 32`, `nt=120`;
- output root changed to a size-specific folder to avoid reusing stale observed
  data from the previous `200 x 88`, `nt=3000` default.

This is benchmark infrastructure only. The production propagator is unchanged.

## 4. Test

Command:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_experimental_fwi_loop_compare.py \
  --device npu:0 \
  --dtype float32 \
  --candidate-mode experimental \
  --result-json docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_experimental_fwi_loop_compare_3shot3iter_20260604.json
```

Result file:

`develope/acoustic_experimental_fwi_loop_compare_3shot3iter_20260604.json`

## 5. Result

| Metric | Production | Experimental |
| --- | ---: | ---: |
| losses | `[0.00019976586918346584, NaN, NaN]` | `[0.00019976586918346584, NaN, NaN]` |
| all finite | false | false |
| peak memory MiB | 53.17578125 | 126.0888671875 |

Other measured values:

| Metric | Value |
| --- | ---: |
| total speedup | 1.852606813950458x |
| forward speedup | 0.678239517849284x |
| backward speedup | 2.2330210932190586x |
| memory ratio | 2.371170939543084x |
| loss diffs all finite | false |

## 6. Interpretation

- The short-loop observed-pressure gate is not valid as an acceptance test in
  this reduced configuration, because production and experimental both become
  non-finite after the first optimizer step.
- The experimental path is not uniquely responsible for the NaN loss trajectory.
- The speedup number is not an acceptance result because the loop is non-finite.
- The candidate memory ratio is higher than production.

## 7. Decision

Stop adding new test-standard work. The branch has enough gates to avoid
misclassification:

- tiny/reduced parity;
- synthetic-energy gradient parity;
- observed-pressure output/loss parity;
- short-loop diagnostic result.

Next work should move to an actual algorithm/operator implementation path, or
stop the current experimental Python prototype if the goal is immediate
production usefulness. Do not run full 40-shot validation for this prototype
until a finite short loop is available.
