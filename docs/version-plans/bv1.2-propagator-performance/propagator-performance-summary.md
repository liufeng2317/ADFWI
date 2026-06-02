# Propagator Performance Summary

Date: 2026-06-02

This file replaces the long sequence of per-round notes as the active summary.
The detailed history is archived under `archive/`.

## Accepted Production Changes

| Change | Scope | Result |
| --- | --- | --- |
| `checkpoint_segments == 1` bypass | default acoustic path | removes unnecessary checkpoint overhead when no segmentation is requested; parity checks were exact |
| source index hoist | default acoustic path | small but valid timestep-loop cleanup; parity checks were exact |

These are the only default acoustic propagator performance changes accepted so
far.

## Accepted Opt-In Paths

| Path | Use case | Decision |
| --- | --- | --- |
| `save_forward_wavefield=False` | FWI runs that do not need forward illumination | valid guarded option; default output contract unchanged |
| `use_custom_chunk_backward=True` | high-memory expert runs | can improve runtime in selected checkpointed cases, but memory growth is large; not a default |

## Closed Or Not Promoted

| Line | Decision | Reason |
| --- | --- | --- |
| receiver stack rewrite | closed | isolated improvement did not transfer to reduced FWI |
| Python pressure expression rewrites | closed | gains were too small or variants were slower |
| `torch.compile` | closed for current NPU env | practical route blocked by missing/unsupported compiler stack |
| `TorchGradProcessor` as default | not promoted | reduced case improved, full-record case did not |
| rematerialized custom checkpoint | not promoted | small end-to-end gain with significant memory increase |
| experimental custom chunk as default | not promoted | useful as expert opt-in only; not a general checkpoint replacement |
| Ascend custom op pressure kernel | paused | single-block pressure/copy can pass, but multi-block copy fails parity |
| no-checkpoint direct return | closed | output/loss/gradient parity stayed exact, but NPU timing slightly worsened |

## Ascend Custom-Op Status

The Ascend route is not a current production optimization path.

What passed:

- scaffold generation,
- package compile,
- runtime visibility through ACLNN symbols,
- PyTorch wrapper build,
- `copy + block_dim=1` exact parity,
- `pressure + block_dim=1` float32-level parity.

What failed:

- `copy + block_dim=8`,
- `copy_vector + block_dim=8`, even with aligned shape.

Decision:

```text
Pause Ascend custom-op production work until the multi-block launch/copy
contract is solved in a focused external custom-op test.
```

## Current Performance Work Boundary

Continue only with production acoustic PyTorch hot-path optimizations that can
be validated by:

1. forward waveform comparison,
2. loss comparison,
3. raw `vp` gradient comparison,
4. seconds/iteration comparison on a named case.

Each future round should make one bounded change, record the result, and stop.
