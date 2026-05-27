# 87. Marmousi2 Benchmark Direction Adjustment

## Purpose

Stop the shot-count sweep after the 5-shot checkpoint-10 gate. The current goal
is no longer to locate the exact NPU throughput or memory knee. The existing
3-shot and 5-shot full-length Marmousi2 gates are sufficient as practical
baselines for near-term optimization work.

## Current Baselines

| Baseline | Purpose | Key result |
| --- | --- | --- |
| 3 shots, 10 iterations, `checkpoint_segments=10` | Fastest current full-case per-iteration NPU baseline | `31.358286083862186s/iteration`, monotonic loss |
| 5 shots, 10 iterations, `checkpoint_segments=10` | Current throughput stress baseline | `31.861114338412882s/iteration`, monotonic loss |

## Decision

- Do not continue to 7-shot or larger shot-count tests for now.
- Avoid spending runtime on locating the exact NPU memory/runtime knee.
- Keep the 3-shot gate for quick full-case performance comparisons.
- Keep the 5-shot gate for heavier throughput validation when an optimization
  plausibly changes batching, memory use, or NPU utilization.

## Updated Optimization Path

The next optimization work should use the existing baselines to validate real
code or workflow changes:

1. improve benchmark/report tooling so repeated full-case runs can be compared
   without manual JSON parsing;
2. evaluate convergence behavior with the existing 3-shot or 5-shot baseline if
   inversion behavior, not raw throughput, is the question;
3. return to code-level optimization only when a concrete bottleneck is visible
   in the fixed baselines.

## Validation

This is a direction and documentation update only. No FWI core code changed, so
no numerical regression test was required for this record. The relevant
validation data are the previously recorded full-case runs in records 85 and 86.
