# Deepwave-Inspired Performance Test Matrix

This matrix defines the validation required before implementing or promoting
any Deepwave-inspired propagator change.

## 1. Reference Baselines

| Baseline | Purpose |
| --- | --- |
| ADFWI production, `checkpoint_segments=10` | current memory-oriented production baseline |
| ADFWI production, `checkpoint_segments=1` | speed and memory upper-bound reference |
| ADFWI opt-in `storage_policy="pressure_remat"` | current accepted expert path |
| full-record Marmousi2 40-shot 10-iteration case | real-case behavior baseline |

Current measured production matrix:

| Shots | Iterations | Checkpoints | Purpose |
| --- | --- | --- | --- |
| 1 | 3, 10 | 10 | single-shot development scale and timing stability |
| 3 | 3, 10 | 10 | small multi-shot batch behavior |
| 40 | 3, 10 | 10 | full-record real-case speed and memory gate |

The measured table is stored in `baselines/baseline-matrix-results.md`.

Checkpoint sweep results are stored in `baselines/checkpoint-sweep-results.md`.
The current observation is:

| Checkpoints | Result |
| --- | --- |
| 10 | current memory-oriented production baseline |
| 5 | slower than 10 and uses more memory in the measured Marmousi2 cases |
| 1 | fastest reference, but peak memory rises by about 6.82x-8.91x |

This means the next high-value optimization target is not simply lowering the
checkpoint count. It should aim for checkpoint=1-like speed while preserving
checkpoint=10-like memory behavior.

## 2. Operator-Level Tests

| Test | Shape/case | Required comparison |
| --- | --- | --- |
| Forward receiver parity | small synthetic acoustic case | `rcv_p`, optional `rcv_u/rcv_w` |
| Forward final state parity | small synthetic acoustic case | final `p/u/w` if exposed |
| Source injection parity | single source and encoded source | receiver output and source index behavior |
| Free-surface parity | free-surface on/off | receiver output |
| PML parity | existing PML parameters | receiver output |

## 3. Gradient Tests

| Test | Requirement |
| --- | --- |
| Raw `vp.grad` parity | compare production autograd and candidate adjoint |
| finite gradient check | no NaN/Inf in raw and processed gradients |
| update norm parity | optimizer update norm matches for short FWI |
| loss trajectory parity | 5/10 iterations match within tolerance |

## 4. Performance Tests

| Test | Required metrics |
| --- | --- |
| one-iteration operator gate | forward time, backward time, total time, peak memory |
| reduced FWI loop | seconds/iteration, loss trajectory, gradient finite |
| full-record 40-shot 10-iteration | mean iteration time, total speedup, peak memory ratio |
| memory policy sweep | device/remat/checkpoint/compressed candidates |

Additional baselines to add only when needed:

| Baseline | When to run |
| --- | --- |
| production `checkpoint_segments=1` | when estimating speed and memory upper bounds |
| current `storage_policy="pressure_remat"` | when comparing against the accepted expert opt-in path |
| reduced/tiny smoke run | during tight implementation loops where full-record matrix is too slow |

## 5. Promotion Rules

| Promotion level | Required evidence |
| --- | --- |
| prototype only | forward parity and clear performance hypothesis |
| benchmark candidate | forward parity, raw gradient finite, one-iteration speed/memory result |
| expert opt-in | short FWI loss/update parity and real-case timing improvement |
| production default | repeated full real-case validation, stable memory, no API confusion |

## 6. Initial Commands To Reuse Or Extend

Existing ADFWI benchmark scripts should be reused before new scripts are added:

| Script | Role |
| --- | --- |
| `scripts/benchmark/acoustic_fwi_iteration_profile.py` | FWI iteration profiling |
| `scripts/benchmark/acoustic_checkpoint_memory_matrix.py` | checkpoint/storage speed-memory matrix |
| `scripts/benchmark/acoustic_experimental_fwi_loop_compare.py` | production vs opt-in FWI comparison |
| `scripts/benchmark/acoustic_memory_phase_breakdown.py` | phase-level memory attribution |

New scripts should be added only when the existing ones cannot express the new
operator contract.
