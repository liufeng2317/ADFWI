# Propagator Performance Change Log

This is the only active modification record for
`bv1.2-propagator-performance`. Detailed per-round notes and raw outputs are in
`archive/` and should not be used as the active plan.

## Accepted Production Changes

| Date | Change | Files | Validation | Decision |
| --- | --- | --- | --- | --- |
| 2026-05-30 | `checkpoint_segments == 1` checkpoint bypass | `ADFWI/propagator/acoustic_kernels.py` | acoustic output/loss/`vp.grad` parity exact on checkpoint-overhead probe | accepted |
| 2026-05-31 | acoustic source-index hoist | `ADFWI/propagator/acoustic_kernels.py` | parity exact; small loop cleanup | accepted |
| 2026-06-02 | skip detached illumination summaries during checkpoint replay | `ADFWI/propagator/acoustic_kernels.py` | reduced/full-record checkpoint=10 loss trajectories matched exactly; steady-state total iteration improved 3.52% reduced and 2.81% full-record | accepted |
| 2026-06-02 | AcousticFWI `pressure_only="auto"` default | `ADFWI/fwi/acoustic_fwi.py`, `scripts/benchmark/acoustic_fwi_iteration_profile.py` | reduced/full-record checkpoint=10 loss trajectories matched exactly; steady-state total iteration improved 9.74% reduced and 10.33% full-record versus forced full-output FWI | accepted |

## Accepted Opt-In Changes

| Date | Change | Files | Validation | Decision |
| --- | --- | --- | --- | --- |
| 2026-05-31 | `save_forward_wavefield=False` guarded output policy | `ADFWI/propagator`, `ADFWI/fwi` | output/loss/raw-gradient parity when illumination is off; guard rejects incompatible illumination use | accepted opt-in |
| 2026-05-31 | `use_custom_chunk_backward=True` expert path | `ADFWI/propagator/acoustic_custom_kernels.py`, acoustic wrapper/FWI plumbing | custom chunk receiver-loss and `vp.grad` parity tests; reduced 5-iteration FWI loss and update matched exactly; total speedup `1.5367x`, backward speedup `1.9362x`, peak allocation `28.9259x` | accepted opt-in, not default; next work is memory reduction |
| 2026-06-02 | `pressure_only=True` acoustic FWI path | `ADFWI/propagator/acoustic_kernels.py`, acoustic wrapper/FWI plumbing | `p`, `forward_wavefield_p`, loss, and `vp.grad` exact parity; reduced checkpoint=10 FWI 5-iteration loss trajectory matched default exactly, steady-state total iteration improved from 29.31s to 25.37s; full-record checkpoint=10 FWI 3-iteration loss trajectory also matched exactly, steady-state total iteration improved from 28.20s to 26.51s | accepted opt-in, not default |

## Accepted Experimental Changes

| Date | Change | Files | Validation | Decision |
| --- | --- | --- | --- | --- |
| 2026-06-02 | forward-saved boundary cache for rematerialized custom chunks | `ADFWI/propagator/acoustic_custom_kernels.py` | reduced 5-iteration FWI loss and update matched exactly; total speedup vs production `1.1312x`; peak allocation `527.9731 MiB` (`1.9374x` production) | accepted for experimental benchmark path only; not enough speed for production promotion |
| 2026-06-02 | rematerialized benchmark default uses every-step divergence cache and block boundary cache | `scripts/benchmark/acoustic_experimental_fwi_loop_compare.py` | reduced 5-iteration FWI loss and update matched exactly; total speedup vs production improved to `1.1989x`, backward speedup `1.3619x`; peak allocation stayed `527.9731 MiB` | accepted benchmark configuration; still not enough speed for production promotion |
| 2026-06-02 | saved-state custom divergence-save candidates | `ADFWI/propagator/acoustic_custom_kernels.py`, benchmark scripts | reduced 5-iteration FWI loss and update matched exactly for compressed candidates; best point saves only `div_p`: total speedup `1.3191x`, backward speedup `1.5719x`, peak allocation `5643.8008 MiB` | accepted benchmark candidates; not enough memory reduction for production promotion |
| 2026-06-02 | rematerialized pressure-only receiver recording | `ADFWI/propagator/acoustic_custom_kernels.py`, benchmark scripts | reduced 5-iteration FWI loss and update matched exactly; total speedup improved to `1.2773x`, backward speedup `1.4484x`; peak allocation stayed `527.9731 MiB` | accepted benchmark candidate; current remat main-line gate |
| 2026-06-02 | rematerialized pressure-only skips velocity receiver chunk copies | `ADFWI/propagator/acoustic_custom_kernels.py` | reduced 5-iteration FWI loss and update matched exactly; candidate absolute total time improved from `111.2747 s` to `106.6786 s`; same-run total speedup `1.2332x`, backward speedup `1.3994x`; peak allocation `528.1079 MiB` | accepted benchmark cleanup; does not change main promotion decision |

## Closed Or Not Promoted

| Date | Line | Result | Decision |
| --- | --- | --- | --- |
| 2026-05-31 | receiver stack rewrite | isolated improvement did not transfer to reduced FWI | closed |
| 2026-05-31 | pressure expression rewrites | too small or slower | closed |
| 2026-05-31 | `torch.compile` | current NPU/compiler environment blocked practical use | closed |
| 2026-05-31 | `TorchGradProcessor` as default | reduced case improved, full-record case did not | not promoted |
| 2026-06-01 | rematerialized custom checkpoint | small end-to-end gain with significant memory increase | not promoted |
| 2026-06-01 | Ascend custom-op pressure kernel | single-block parity passed, multi-block copy failed | paused |
| 2026-06-02 | no-checkpoint direct return | parity exact, NPU timing slightly worse | reverted and closed |
| 2026-06-02 | non-reentrant PyTorch checkpoint for `checkpoint_segments > 1` | reduced checkpoint=10 baseline ran, but non-reentrant candidate failed during NPU TorchScript backward recompute | reverted and closed |
| 2026-06-02 | skip detached forward-wavefield summaries during checkpoint replay | checkpoint=10 backward improved only slightly, while checkpoint=1 path regressed due TorchScript signature/branch overhead | reverted and closed |
| 2026-06-02 | detach-before-summary accumulation | backward improved only slightly, forward regressed, and steady-state total iteration improved only 0.64% | reverted and closed |
| 2026-06-02 | empty placeholders for skipped replay summaries | preserved loss/gradient behavior but slowed reduced checkpoint=10 steady-state total iteration by 2.10% | reverted and closed |
| 2026-06-02 | concatenate segmented receiver chunks | preserved loss/gradient behavior but slowed reduced checkpoint=10 steady-state total iteration by 1.93% | reverted and closed |

## Readability And Cleanup

| Date | Change | Files | Validation | Decision |
| --- | --- | --- | --- | --- |
| 2026-06-02 | clarified custom-kernel internal names and removed stale acoustic debug remnants | `ADFWI/propagator/acoustic_custom_kernels.py`, `ADFWI/propagator/acoustic_kernels.py`, `ADFWI/propagator/acoustic_propagator.py` | `py_compile`; custom chunk backend integration tests passed | accepted cleanup |

## Current Remaining Boundary

| Item | Status | Next action |
| --- | --- | --- |
| `ADFWI/propagator/acoustic_custom_kernels.py` | valid expert opt-in / benchmark path | keep, but do not make default without new full FWI evidence |
| saved-state custom chunk memory | peak allocation rises `28.9259x` in reduced FWI comparison | optimize memory only if loss/update parity and core backward speed benefit are preserved |
| saved-state divergence compression | best current point still uses `20.7x` production peak memory | do not promote; next valuable work needs structural state compression, not more divergence-component sweeps |
| rematerialized custom chunk speed | peak allocation is controlled and current pressure-only candidate total time is `106.6786 s` for 5 reduced iterations | continue only with changes that reduce rematerialized backward replay cost; receiver-output cleanup is now mostly exhausted |
| `ADFWI/propagator/acoustic_kernels_bs.py` | research prototype, not production-wired | archive or delete only in a separate explicit cleanup |
| Ascend custom-op route | paused | resume only after standalone multi-block copy parity is solved |
