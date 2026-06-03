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
| 2026-06-02 | lazy zero placeholders for acoustic pressure-only outputs | `ADFWI/propagator/acoustic_kernels.py` | pressure-only loss and `vp.grad` parity test passed; reduced checkpoint=10 10-iteration loss trajectory and `vp_update_norm` matched baseline; steady-state total iteration improved from `26.4793 s` to `25.9895 s` | accepted |

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
| 2026-06-02 | remat every-step divergence fast path | reduced 5-iteration FWI loss and update matched exactly, but candidate total time regressed versus the current best remat candidate (`116.2597 s` vs `106.6786 s`) | reverted and closed |
| 2026-06-02 | omit pressure-only velocity placeholders from internal FWI record | reduced checkpoint=10 10-iteration loss trajectory and `vp_update_norm` matched exactly, but steady-state total iteration regressed from `25.9895 s` to `26.5913 s` | reverted and closed |
| 2026-06-02 | hoist pressure-only source-shape checks outside the time loop | reduced checkpoint=10 10-iteration loss trajectory and `vp_update_norm` matched exactly, but steady-state total iteration regressed from `25.9895 s` to `26.2773 s` | reverted and closed |

## Readability And Cleanup

| Date | Change | Files | Validation | Decision |
| --- | --- | --- | --- | --- |
| 2026-06-02 | clarified custom-kernel internal names and removed stale acoustic debug remnants | `ADFWI/propagator/acoustic_custom_kernels.py`, `ADFWI/propagator/acoustic_kernels.py`, `ADFWI/propagator/acoustic_propagator.py` | `py_compile`; custom chunk backend integration tests passed | accepted cleanup |

## Diagnostic Records

| Date | Diagnostic | Files | Result | Decision |
| --- | --- | --- | --- | --- |
| 2026-06-02 | pressure-only acoustic backward operator profile | `scripts/benchmark/acoustic_backward_operator_profile.py`, `acoustic_pressure_only_backward_operator_profile_20260602.json` | `checkpoint_segments=10`, `pressure_only=True`, `nt=400`; backward `21.6333 s`; top self-device events are `aten::copy_`, `aten::slice`, `CheckpointFunctionBackward`, `aten::slice_backward`, `empty_tensor`, `aclnnInplaceCopy`, `SliceBackward0`, and zero/allocation ops | use this as the gate for the next route: stop receiver/output micro-cleanups and only continue with a bounded prototype that reduces autograd slice-assignment graph cost |
| 2026-06-02 | pressure inner-state recurrence prototype | `scripts/benchmark/acoustic_pressure_inner_state_microbenchmark.py`, `acoustic_pressure_inner_state_microbenchmark_20260602.json` | fixed the standalone reference so it is autograd-valid; candidate keeps only pressure interior as recurrent state; output/loss and `p/u_seq/w_seq/kappa1/alpha1` gradients match exactly; mean total speedup `1.1714x` | continue this line only by extending it to coupled `p/u/w` recurrence; do not edit production kernel from this pressure-only subproblem alone |
| 2026-06-03 | coupled `p/u/w` custom-backward pressure-loss probe | `scripts/benchmark/acoustic_custom_multistep_update_probe.py`, `acoustic_custom_multistep_pressure_loss_probe_repeat_20260603.json` | source, free surface, receiver recording enabled; pressure-loss components `p,rcv_p`; output/loss diff `0.0`; max gradient abs diff `1.19e-7`; mean backward speedup `1.9633x`; mean total speedup `1.5144x` | strongest current evidence for the next high-value route; continue with longer-step or production-chunk gate before touching production kernel |
| 2026-06-03 | longer-step coupled custom-backward pressure-loss gate | `scripts/benchmark/acoustic_custom_multistep_update_probe.py`, `acoustic_custom_multistep_pressure_loss_steps80_20260603.json` | same source/free-surface/receiver pressure-loss gate at `steps=80`; output/loss diff `0.0`; max gradient abs diff `4.77e-7`; mean backward speedup `2.0144x`; mean total speedup `1.5444x` | custom-backward route remains stable at longer recurrence length; next gate must use production acoustic input preparation and raw `vp.grad` comparison |
| 2026-06-03 | production-interface experimental chunk gates | `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`, `acoustic_production_chunk_*_20260603.json` | observed-pressure `nt=120/240` had exact output/loss parity but both production and candidate raw `vp.grad` were non-finite; synthetic-energy `nt=120` had output/loss diff `0.0`, finite raw gradients, raw `vp.grad` max abs diff `7.11e-15`, backward speedup `1.9353x`, total speedup `1.4474x` | custom chunk is valid under finite synthetic-energy production input, but pressure-loss promotion still needs a finite observed-pressure gate |
| 2026-06-03 | reduced-shape finite observed-pressure production-chunk gate | `scripts/benchmark/acoustic_experimental_forward_iteration_parity.py`, `acoustic_production_chunk_observed_reduced_nt3000_20260603.json` | `nx=200`, `nz=88`, `nt=3000`; output/loss diff `0.0`; reference and candidate raw `vp.grad` finite; raw `vp.grad` max abs diff `2.80e-7`; backward speedup `1.8051x`; total speedup `1.3829x`; peak memory ratio `55.0816x` | speed route passes the finite pressure-loss gate, but saved-state memory is too high; next work must reduce memory for this path |
| 2026-06-03 | checkpoint=1 speed/memory upper-bound matrix | `scripts/benchmark/acoustic_checkpoint_memory_matrix.py`, `acoustic_checkpoint_memory_matrix_observed_reduced_20260603.json` | same reduced observed-pressure gate: production checkpoint=1 total speedup `1.2087x`, peak memory `7.4425x`; production custom chunk total speedup `1.3435x`, peak memory `27.1292x`; loss/output parity exact and raw `vp.grad` finite | both fail the new `<=2.5x` memory constraint; keep them as speed upper-bound references only, and continue with memory-bounded rematerialized/custom backward |
| 2026-06-03 | memory-budget remat cache policy gate | `scripts/benchmark/acoustic_checkpoint_memory_matrix.py`, `acoustic_remat_pressure_*_gate_20260603.json` | same reduced observed-pressure gate; full divergence cache speedup `1.2049x` but memory `2.7631x` fails; no divergence cache speedup `1.0390x`, memory `1.7192x`; `div_p` cache speedup `1.0654x`, memory `2.0635x`; all loss diffs `0.0`, raw `vp.grad` finite | use `div_p`-only remat pressure candidate as the current budget-valid baseline; next code-level work must improve this path without exceeding `2.5x` memory |
| 2026-06-03 | scripted pressure-only remat forward | `ADFWI/propagator/acoustic_custom_kernels.py`, `scripts/benchmark/acoustic_chunk_forward_overhead.py`, `acoustic_remat_pressure_scripted_*_20260603.json` | pressure-only remat forward moved from Python list/stack loop to TorchScript helper; forward-only candidate changed from `5.4453 s` to `3.4458 s`; full observed-pressure gate total speedup improved from `1.0654x` to `1.1752x`; peak memory stayed `2.0635x`; loss diff `0.0`, raw `vp.grad` finite | accepted budget-valid optimization; next work should target backward replay cost, not forward wrapper overhead |
| 2026-06-03 | targeted backward operator profiler feasibility check | no code retained | observed-pressure `nt=800` profile was invalid because production raw `vp.grad` became non-finite; finite `nt=3000` and tiny synthetic-energy profiler runs were too slow for iteration work on the current NPU/custom-autograd path | pause autograd profiler for this route; use lightweight staged timing or formula-level replay analysis instead |
| 2026-06-03 | lightweight remat backward stage timing | `ADFWI/propagator/acoustic_custom_kernels.py`, `scripts/benchmark/acoustic_remat_backward_stage_timing.py`, `acoustic_remat_backward_stage_timing_divp_20260603.json` | current budget-valid remat path: replay states/divergence `5.3446 s` (`27.34%` backward), gradient-buffer init `0.0020 s`, reverse adjoint loop `14.1475 s` (`72.36%` backward); loss finite and raw `vp.grad` finite | next optimization should target reverse adjoint loop, not replay construction |

## Current Remaining Boundary

| Item | Status | Next action |
| --- | --- | --- |
| `ADFWI/propagator/acoustic_custom_kernels.py` | valid expert opt-in / benchmark path | keep, but do not make default without new full FWI evidence |
| saved-state custom chunk memory | peak allocation rises `28.9259x` in reduced FWI comparison | outside the `<=2.5x` memory budget; keep only as speed-ceiling reference |
| saved-state divergence compression | best current point still uses `20.7x` production peak memory | do not promote; next valuable work needs structural state compression, not more divergence-component sweeps |
| rematerialized custom chunk speed | peak allocation is controlled and current pressure-only candidate total time is `106.6786 s` for 5 reduced iterations | continue only with changes that reduce rematerialized backward replay cost; receiver-output cleanup is now mostly exhausted |
| `ADFWI/propagator/acoustic_kernels_bs.py` | research prototype, not production-wired | archive or delete only in a separate explicit cleanup |
| Ascend custom-op route | paused | resume only after standalone multi-block copy parity is solved |
