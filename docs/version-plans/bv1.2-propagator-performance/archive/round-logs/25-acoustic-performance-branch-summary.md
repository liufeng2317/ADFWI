# 25 - Acoustic Performance Branch Summary

Date: 2026-05-31

## Purpose

Summarize the acoustic performance work so far and keep the branch focused on
measured performance optimization. This branch remains the performance branch;
future high-impact experiments continue here, but they must be gated by strict
numerical validation.

## Accepted Changes

| Change | Status | Result |
| --- | --- | --- |
| `checkpoint_segments == 1` bypasses checkpoint wrapper | accepted default path | removes unnecessary checkpoint overhead when no segmentation is requested; numerical differences `0.0` in the benchmark |
| acoustic source index hoist | accepted default path | small speedup with numerical differences `0.0` |
| `save_forward_wavefield=False` when illumination is disabled | accepted opt-in | repeated reduced FWI comparison preserved loss/grad/update exactly; total speedup mean `1.16x` |
| validation scripts expose `--gradient-processor legacy|torch` | accepted opt-in | keeps default legacy path while allowing NPU experiments |

## Rejected Or Deferred Routes

| Route | Decision | Reason |
| --- | --- | --- |
| `torch.cat` functional pressure reconstruction | rejected | exact but slower |
| receiver list/stack recording | rejected | isolated faster but reduced FWI slower |
| checkpoint segmentation as speed path | rejected | useful for memory, slower when memory is sufficient |
| `torch.compile` on current NPU environment | rejected | default optimizing backend blocked by missing `triton`; eager backend not useful |
| Python-level pressure inner-state recurrence | rejected | hit autograd in-place version constraints |
| `TorchGradProcessor` as full-record default | rejected | reduced case faster, full-record 10-iteration run slower |

## Current Bottleneck

Full-record acoustic iteration profile:

| Component | Seconds | Share |
| --- | ---: | ---: |
| backward | `16.93 s` | `59.86%` |
| forward | `9.83 s` | `34.78%` |
| gradient processing | `0.69 s` | `2.44%` |
| optimizer step | `0.78 s` | `2.77%` |

Full-record backward operator profile confirms the same dominant families as
the small profile:

- `slice_backward` / `SliceBackward0`;
- `copy_` / `InplaceCopy`;
- `zero_` / `zeros` / `empty_tensor`;
- `CopySlices` / index-put work.

## Branch Direction

The next useful work is no longer another profile and no longer another small
mechanical rewrite. The remaining high-impact route is:

```text
custom-gradient / adjoint-style acoustic update work, starting from a tiny
validated prototype and scaling only if output and gradient parity hold.
```

This remains on `bv1.2-propagator-performance`; no new branch is required.

## Immediate Next Task

Start with a minimal custom-autograd pressure-update probe:

- compare one acoustic pressure interior update against normal PyTorch
  autograd;
- check output, loss, and gradients for `p`, `u`, `w`, `kappa`, and `alpha`;
- measure forward/backward wall time;
- do not touch production propagator code.

If this prototype cannot preserve gradients or does not show a meaningful
backward speed signal, stop custom-autograd work before attempting a full
kernel rewrite.
