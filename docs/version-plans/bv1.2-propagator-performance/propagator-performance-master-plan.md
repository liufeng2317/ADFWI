# Propagator Performance Master Plan

Date: 2026-05-31

This document is the execution entry point for the
`bv1.2-propagator-performance` branch. It replaces ad-hoc optimization with a
fixed profiling and validation route.

## Goal

Improve `ADFWI/propagator` performance without changing the default numerical
results, automatic-differentiation gradients, or public output contract.

The optimization target is not "cleaner code" by itself. The target is measured
runtime reduction on a named FWI workflow while preserving scientific behavior.

## Non-Negotiable Rules

1. Profile before editing kernels.
2. Keep default numerical behavior unchanged.
3. Compare gradients for every differentiable propagator change.
4. Treat opt-in output reduction as a separate feature from default
   performance optimization.
5. Stop a phase when its exit condition is reached.
6. Do not start elastic optimization until the acoustic profiling workflow has
   converged and been recorded.

## Baselines

### B0: Stable Full-Record Baseline

Reference:
`docs/version-plans/bv1.2/full-record-marmousi2-baseline.md`

| Metric | Value |
| --- | --- |
| Case | `examples/validation/marmousi2_acoustic_full_record` |
| Device / dtype | `npu:0`, `float32` |
| `checkpoint_segments` | `1` |
| Forward shape | `[40, 3000, 200]` |
| Forward pressure norm | `4.604166507720947` |
| Forward wall time | `7.913247490301728 s` |
| 300-iter final loss | `4211.63671875` |
| 300-iter min loss | `2709.09228515625` |
| Seconds / iteration | `33.75309997430071 s` |

### B1: Current Acoustic Kernel Baseline

This branch has already accepted two default acoustic kernel optimizations:

| Change | Effect | Numerical result |
| --- | --- | --- |
| Bypass checkpoint when `checkpoint_segments == 1` | small NPU benchmark total mean `9.40 s -> 6.86 s` | `p/u/w`, forward wavefields, loss, and `vp.grad` differences all `0.0` |
| Hoist acoustic source indices out of timestep loop | small NPU benchmark total mean `6.86 s -> 6.55 s` | same numerical parity, differences all `0.0` |

The next default optimization must compare against the current branch, not the
pre-optimization code.

### B2: Opt-In Output Policy Baseline

`save_forward_wavefield=False` is an opt-in acoustic path guarded at FWI level.
It is valid only when active gradient processors have
`forw_illumination=False`.

This is not the default behavior baseline. Do not mix opt-in output reduction
with default kernel optimization in the same performance conclusion.

## Execution Route

```text
Phase A: End-to-end cost breakdown
  -> Phase B: Acoustic AD graph and backward cost
  -> Phase C: Acoustic timestep/kernel execution cost
  -> Phase D: Acoustic FWI loop and gradient-processing cost
  -> Phase E: Elastic propagator profiling
  -> Phase F: Research-only high-risk methods
```

Only one phase is active at a time. Each phase starts with a measurement record
and ends with a decision: implement one bounded change, defer it, or stop.

## Phase A: End-To-End Cost Breakdown

Purpose: determine where runtime is actually spent before more kernel edits.

Required measurement:

- one reduced differentiable acoustic FWI iteration;
- one full-record acoustic forward;
- timing split for setup, `propagator.forward`, loss preparation, loss
  evaluation, backward, gradient processing, optimizer step, and output/save
  overhead;
- device, dtype, shape, shot count, receiver count, `nt`, and
  `checkpoint_segments`.

Exit condition:

- a recorded table that identifies the top two runtime components;
- a selected Phase B/C/D target with expected impact and validation commands;
- no code optimization before this table exists.

Next immediate task:

```text
Build an acoustic FWI iteration profiling harness that measures the runtime
components above on the reduced Marmousi2 validation case.
```

## Phase B: Acoustic AD Graph And Backward Cost

Purpose: reduce automatic-differentiation overhead without changing gradients.

Candidate methods:

- checkpoint/rematerialization policy when memory permits;
- graph pruning for outputs that are not used by the loss or gradient
  processors;
- avoiding unnecessary differentiable tensor writes when the value is not used;
- only consider custom autograd/adjoint-state after default PyTorch AD
  bottlenecks are quantified.

Required validation:

- forward output parity for `p`, `u`, `w`;
- forward-wavefield parity when default outputs are enabled;
- loss parity;
- raw `vp.grad` parity;
- processed gradient parity if FWI code is touched;
- reduced inversion comparison.

Exit condition:

- one accepted bounded improvement, or a record that AD graph cost is not the
  next dominant bottleneck.

## Phase C: Acoustic Timestep And Kernel Execution Cost

Purpose: optimize measured per-timestep overhead in the acoustic kernel.

Candidate methods:

- move invariant index/tensor preparation out of timestep loops;
- reduce repeated tiny allocations when safe for autograd;
- receiver sampling cost reduction only if receiver sampling is measured as
  visible;
- wavefield accumulation cost reduction only behind default-parity or explicit
  opt-in policy;
- device-specific compile/fusion experiments only after normal PyTorch timing
  is stable.

Required validation:

- same as Phase B for any differentiable kernel edit;
- full-record forward timing for accepted milestone changes.

Exit condition:

- stop when remaining measured candidates are below `5%` reduced-iteration
  impact or below `10%` isolated hot-path impact.

## Phase D: Acoustic FWI Loop And Gradient Processing Cost

Purpose: optimize work outside the propagation timestep loop only if profiling
shows it matters.

Candidate methods:

- compare `GradProcessor` and `TorchGradProcessor` on the same gradient;
- identify CPU/NPU transfer points;
- reduce repeated conversion, smoothing, and normalization setup;
- separate plotting/save overhead from compute overhead.

Required validation:

- `tests/test_torch_grad_processor.py`;
- processed-gradient numerical comparison;
- reduced inversion loss comparison.

Exit condition:

- one accepted loop-level improvement, or a record that loop overhead is not
  significant relative to propagation/backward.

## Phase E: Elastic Propagator Profiling

Purpose: start elastic optimization only after acoustic has a stable
measurement/validation workflow.

Required measurement:

- PML vs ABL branch timing;
- `fd_order` 4/6/8/10 timing when practical;
- receiver output and forward-wavefield output cost;
- backward/gradient comparison for a tiny differentiable elastic case.

Exit condition:

- an elastic-specific optimization plan based on measurements, not on file
  size or duplicated code alone.

## Phase F: Research-Only High-Risk Methods

These methods are not default branch tasks until a separate experiment proves
both speed and scientific parity.

| Method | Reason for caution |
| --- | --- |
| custom autograd / adjoint-state rewrite | high risk of gradient convention or boundary mismatch |
| mixed precision / AMP | may alter inversion trajectory and stability |
| `torch.compile` or graph capture | may be backend/version sensitive, especially on NPU |
| replacing checkpoint strategy globally | can change memory use and backward behavior |
| promoting `acoustic_kernels_bs.py` | currently experimental and has global autograd side effects |

Current acoustic custom-gradient status:

```text
The benchmark-only custom recurrence is not production-ready. Direct replay of
the exact observed-pressure receiver upstream preserved outputs/loss exactly
but failed raw `vp.grad` parity (`3.63e-4` max abs diff). Further work stayed at
the formula-level backward-validation stage until the smallest local failing
case identified a `gw` view-aliasing bug in the custom backward. After cloning
`gw`, CPU float64 formula-level tests passed to machine precision and the full
observed-upstream direct replay improved to `2.90e-7` raw `vp.grad` max abs
diff. The first reduced validation production-interface parity gate preserved
outputs/loss exactly and kept raw `vp.grad` max abs diff at `2.90e-7`, but the
candidate was slower because it was still a Python-loop prototype. After adding
batched source execution to the benchmark-only prototype, reduced validation
FWI-style parity kept raw `vp.grad` max abs diff at `2.89e-7`, improved backward
by `1.34x`, and improved total measured iteration by `1.04x`. The accepted
production-facing route is a chunk-level custom autograd wrapper; timestep-level
production replacement is rejected because it preserves Python dispatch
overhead. The first chunk-level benchmark-only prototype preserves outputs/loss
exactly, passes a 300-step CPU float64 gradient gate with `1.82e-11` max abs
diff, and gives `1.80x` total speedup on a 300-step NPU float32 probe. Its NPU
float32 raw-gradient difference is `5.86e-3` max abs (`7.40e-4` max rel), so it
must next pass a production-interface validation-geometry parity gate before
any production kernel edit.
```

## Stop Criteria

Stop the current optimization round when any of these is true:

- no measured bottleneck has a plausible `>=5%` reduced-iteration impact;
- the proposed change cannot preserve raw gradient parity;
- the change requires public default output changes;
- the change is only readability cleanup and not tied to measured runtime;
- the same phase has produced two consecutive "no meaningful improvement"
  records.

At that point, write a closeout note and move to the next planned phase or stop
the branch.

## Required Record For Each Change

Each accepted optimization needs a local record in this folder with:

- optimization path and hypothesis;
- pre-change command and timing;
- post-change command and timing;
- shape, device, dtype, and case;
- max absolute and relative differences for outputs and gradients;
- validation commands;
- commit hash;
- next direction.

No propagator performance commit should be merged without this record.

## Current Status After Phase D

Latest records:

- `18-acoustic-phase-d-gradient-processor-profile.md`
- `19-acoustic-phase-d-gradient-processor-10iter.md`
- `20-acoustic-validation-gradient-processor-option.md`
- `21-acoustic-full-record-gradient-processor-10iter.md`
- `22-acoustic-full-record-iteration-profile.md`
- `23-acoustic-full-record-backward-operator-profile.md`
- `24-acoustic-timestep-rewrite-decision.md`
- `25-acoustic-performance-branch-summary.md`
- `26-acoustic-custom-pressure-update-probe.md`
- `27-acoustic-custom-timestep-update-probe.md`
- `28-acoustic-custom-multistep-update-probe.md`
- `29-acoustic-custom-source-freesurface-probe.md`
- `30-acoustic-custom-receiver-recording-probe.md`
- `31-acoustic-custom-kernel-parity-probe.md`
- `32-acoustic-experimental-forward-path.md`
- `33-acoustic-experimental-forward-iteration-parity.md`
- `34-acoustic-gradient-difference-localization.md`
- `35-acoustic-receiver-difference-location.md`
- `36-acoustic-observed-loss-upstream-probe.md`
- `37-acoustic-targeted-backward-exclusions.md`
- `38-acoustic-observed-upstream-direct-replay.md`
- `39-acoustic-observed-scale-local-recurrence.md`
- `40-acoustic-custom-backward-view-alias-fix.md`
- `41-acoustic-production-interface-parity.md`
- `42-acoustic-batch-source-custom-prototype.md`
- `43-acoustic-production-facing-design.md`
- `44-acoustic-custom-chunk-forward-prototype.md`
- `45-acoustic-production-interface-chunk-parity.md`
- `46-acoustic-observed-finite-chunk-gate.md`
- `47-acoustic-chunk-forward-overhead.md`
- `48-acoustic-production-custom-chunk-opt-in.md`
- `49-acoustic-custom-chunk-5iter-fwi-validation.md`
- `50-acoustic-segmented-custom-chunk-gate.md`
- `51-acoustic-segmented-custom-chunk-5iter-fwi.md`
- `52-acoustic-custom-chunk-memory-profile.md`
- `53-acoustic-custom-chunk-full-record-validation.md`
- `54-acoustic-custom-chunk-api-contract.md`
- `55-acoustic-rematerialized-custom-checkpoint-plan.md`
- `56-acoustic-rematerialized-custom-checkpoint-gate.md`
- `57-acoustic-remat-div-cache-memory-tradeoff.md`
- `58-acoustic-remat-div-cache-stride-sweep.md`

Decision:

- `TorchGradProcessor` is numerically stable for the measured acoustic NPU
  cases.
- It remains an explicit opt-in path because full-record 10-iteration timing
  was slower than legacy (`232.00 s -> 240.73 s`).
- Do not continue optimizing Phase D unless a new profile shows gradient
  processing has become dominant.

Next acoustic direction:

```text
Return to propagation/backward cost. The full-record single-iteration profile
measured backward at `16.93 s` (`59.86%`) and forward at `9.83 s` (`34.78%`).
The full-record backward operator profile confirms the earlier small-profile
conclusion: slice/copy/zero/allocation autograd overhead dominates. Stop
repeating backward profile runs. The timestep rewrite decision rejects another
small default-path rewrite in this branch because the remaining meaningful
changes would alter the autograd representation of the recurrent wave equation.
```

Current next direction:

```text
Continue on this branch with custom-gradient acoustic prototypes. The first
pressure-update probe preserved output/loss, kept gradient max absolute
difference at `2.27e-13`, and showed `1.67x` isolated backward speedup. The next
complete one-timestep p/u/w prototype also preserved output/loss, kept gradient
max absolute difference at `1.36e-12`, and showed `1.87x` isolated backward
speedup. The 20-step recurrence prototype preserved final outputs/loss, kept
gradient max absolute difference at `5.68e-14`, and showed `1.87x` backward
speedup. The source/free-surface prototype preserved outputs/loss, kept
gradient max absolute difference at `1.26e-12`, and showed `2.05x` backward
speedup with `1.58x` total speedup. Adding receiver recording preserved final
outputs, recorded traces, and loss exactly, kept gradient max absolute
difference at `4.82e-11`, and showed `1.98x` backward speedup with `1.54x`
total speedup. The first production-interface parity harness against
`forward_kernel` passed on a tiny NPU case: receiver records/loss matched
exactly, `v.grad` maximum absolute difference was `4.14e-25`, and total speedup
was `1.67x`. The custom recurrence is now exposed as a benchmark-only
`experimental_forward_kernel`; the refactored tiny NPU parity kept receiver
records/loss exact, kept `v.grad` maximum absolute difference at `4.14e-25`,
and showed `1.77x` total speedup. The next gate is reduced Marmousi2 iteration
parity with this experimental forward path, still outside default propagator
APIs.

Reduced Marmousi2 FWI-style parity did not pass. Loss matched and receiver
output absolute differences were small, but raw `vp.grad` maximum absolute
difference was `3.63e-4`, which is too large for a core differentiable
propagator change. Stop expanding the experimental path and localize the
gradient difference before considering production integration.

Gradient localization ruled out waveform normalization, NPU repeat
nondeterminism, long `nt=3000` recurrence length, and production batched-source
execution versus experimental per-source looping as primary causes. With the
validation source/survey but synthetic-energy loss, raw gradient difference
dropped to `3.37e-14`. The next active task is receiver-output difference
localization under the validation source/wavelet because observed-pressure
residual loss amplifies a `6.52e-09` receiver-output difference into the failed
raw-gradient parity.

Receiver-difference localization found no remaining receiver-output difference
after matching the production wrapper contract. The previous `6.52e-09`
difference came from the experimental benchmark path skipping
`model.forward()`. After adding the refresh, receiver outputs and loss matched
exactly, but raw `vp.grad` still differed by `3.63e-4`. The active issue is now
custom-backward parity under the observed-pressure loss upstream gradient, not
receiver-output location.

Observed-loss upstream probing confirmed that the loss itself is not the
mismatch source: receiver outputs and receiver upstream gradients are exactly
equal. The same external receiver upstream gradient still yields raw `vp.grad`
max abs diff `3.63e-4`. The active target is custom backward localization under
fixed external upstream gradients.

Targeted exclusions ruled out free-surface adjoint, pressure-only receiver
loss, random pressure upstream, and long-time random pressure upstream as
primary causes. Direct observed-upstream replay then exposed and fixed a
`gw` view-aliasing issue in the custom backward. After that fix,
production-interface parity kept outputs/loss exact and reduced raw `vp.grad`
max abs diff to `2.90e-7`, but the timestep-level prototype was too slow for
production. Batched source execution improved the benchmark-only reduced
validation path to `1.34x` backward speedup and `1.04x` total speedup while
keeping raw `vp.grad` max abs diff at `2.89e-7`.

The active route is now chunk-level custom autograd, not more timestep-level
debugging. The first chunk-level benchmark prototype preserved outputs/loss
exactly, passed the 300-step CPU float64 gradient gate with `1.82e-11` max abs
diff, and measured `1.80x` total speedup on a 300-step NPU float32 probe. The
first production-interface chunk gate passed with synthetic-energy loss:
outputs/loss exact, raw `vp.grad` max abs diff `3.55e-15`, and total speedup
`1.58x`. The observed-pressure reduced gate is currently invalid because
production itself produces non-finite raw gradients in that reduced shape. The
fullshape observed-pressure gate is valid and passes: both production and chunk
gradients are finite, outputs/loss are exact, raw `vp.grad` max abs diff is
`2.89e-7`, backward speedup is `1.22x`, and total speedup is `1.11x`. The next
task was to explain the chunk candidate forward overhead before any production
kernel edit. The forward-only overhead benchmark shows timestep-custom forward
is slower, but chunk-custom forward is comparable to production and averaged
`1.09x` faster across the repeat. The next task is a guarded production
integration design, not more low-level formula debugging. The guarded opt-in
production path now exists as `use_custom_chunk_backward=True`, limited to
`checkpoint_segments == 1` and `save_forward_wavefield=False`. The fullshape
observed-pressure production gate passes with exact receiver outputs/loss, raw
`vp.grad` max abs diff `2.89e-7`, and total speedup `1.22x`. The next task is a
short real FWI validation of the opt-in path before considering broader API
exposure.

The 5-iteration fullshape observed-pressure FWI validation confirms the
single-step gain transfers to a short real loop: loss trajectories match to
`4.88e-4` absolute final-loss difference, all raw/processed gradients and
post-step models are finite, and total compute improves `124.33 s -> 102.32 s`
(`1.215x`). The gain is from backward (`1.329x` mean speedup), while forward is
roughly neutral.

The opt-in path now supports segmented execution through `checkpoint_segments`.
For `checkpoint_segments=10`, the fullshape observed-pressure gate passes with
exact receiver outputs/loss, raw `vp.grad` max abs diff `2.74e-7`, backward
speedup `1.93x`, and total speedup `1.51x`. This is not yet a memory-equivalent
checkpoint replacement because the custom path does not rematerialize states in
backward. The next task is a short FWI loop with `checkpoint_segments=10`, then
peak-memory measurement before any stronger claim.

The 5-iteration `checkpoint_segments=10` FWI validation also passes: loss
trajectory is exactly identical, all finite checks pass, `vp_update_norm`
matches, and total compute improves `131.83 s -> 91.01 s` (`1.449x`). Forward is
slower (`0.652x`), but backward is much faster (`1.844x`), producing a clear net
gain.

Peak-memory profiling resolves the checkpoint-positioning question. On the same
reduced fullshape one-iteration case, the custom segmented path improves total
time `29.22 s -> 20.15 s` (`1.451x`) and backward `23.83 s -> 12.79 s`
(`1.863x`), but peak allocated memory increases from `272.38 MiB` to
`7876.88 MiB` (`28.92x`). Therefore this is an opt-in high-memory acceleration
mode, not a memory-equivalent checkpoint replacement. The next acoustic step is
to document the production API contract for this opt-in path and validate a
short full-record run before considering deeper rematerializing custom backward
work.

The full-record validation confirms that positioning. With all `40` shots in a
single batch, the custom path OOMs on the tested NPU. With the same full-record
data and `batch_size=20`, three FWI iterations pass: loss trajectories are
identical, all finite checks pass, and mean iteration time improves
`55.00 s -> 37.59 s` (`1.463x`). A direct one-iteration tensor parity check
keeps receiver outputs/loss exact and raw `vp.grad` max abs diff at
`4.47e-7`, with total speedup `1.486x`. The path should remain an expert
opt-in high-memory speed mode. Do not make it default. The next task is to
formalize this API contract and then either stop this line or open a separate
true-rematerialization design.

The API contract is now formalized in the acoustic FWI public entry point:
`use_custom_chunk_backward=False` remains the default, while expert users may
opt in with `save_forward_wavefield=False`. The FWI layer and propagator layer
both reject `use_custom_chunk_backward=True` with `save_forward_wavefield=True`
so the path cannot be confused with the default forward-wavefield output
contract. This is a wiring/API change only; no default kernel behavior changes.

The next checkpoint-specific line is rematerialized custom backward for
`checkpoint_segments > 1`. This is distinct from the high-memory custom chunk
path: forward stores only chunk boundary states, and backward recomputes chunk
internals before applying the manual adjoint. The initial implementation is an
experimental kernel only and is not wired into the default propagator path.

The first rematerialized checkpoint gate passes numerically and shows speedup,
but not checkpoint-equivalent memory. On the fullshape reduced observed-pressure
gate, receiver outputs/loss match exactly and raw `vp.grad` max abs diff is
`2.74e-7`; total speed improves `1.229x`. Peak allocated memory is
`810.92 MiB` versus production checkpoint `292.98 MiB` (`2.77x`). This is much
better than the high-memory custom chunk path (`28.92x`), but still not close
enough for default integration. Continue only with memory-focused rematerialized
backward work, not another speed-only benchmark.

The memory-focused div-cache experiment explains the `2.77x` ratio. Removing
persistent `div_p/div_u/div_w` lists from rematerialized backward lowers the
fullshape peak ratio from `2.77x` to `1.72x` while preserving receiver
outputs/loss and raw `vp.grad` max abs diff (`2.74e-7`). The cost is speed:
fullshape total speedup drops from `1.229x` to `1.002x`. The next useful
experiment is not more broad validation; it is a chunk-length/memory-speed sweep
for the rematerialized path, or a lower-level implementation if Python-level
state management cannot keep both memory and speed.

The first div-cache stride sweep found a practical experimental compromise.
Compared with production checkpoint, `stride=2` gives `2.25x` peak memory,
`1.149x` total speedup, and `1.227x` backward speedup while preserving exact
receiver outputs/loss and the same raw `vp.grad` max abs diff (`2.74e-7`).
`stride=4` reaches `1.98x` memory but loses nearly all speedup (`1.001x` total).
For this Python-level rematerialized prototype, `divergence_cache_stride=2` is
the current recommended experimental point. The next gate should be short real
FWI or full-record validation before any production-facing API.

The component-level divergence cache probe closed a tempting but weak subline.
Caching only `p`, `u`, `w`, or `p,u` keeps receiver outputs/loss exact and raw
`vp.grad` max abs diff at `2.74e-7`, but does not keep enough speed. The best
single-component fullshape result after component-aware recompute is only
`1.072x` total speedup at `2.07x` peak memory, and `p,u` is `1.039x` at
`2.42x`. Therefore do not continue component-cache enumeration. The next
memory-focused task is to measure and reduce stored reverse state lists
(`p/u/w` states) while using the current `stride=2` all-component cache as the
speed-preserving reference.

The state-cache probe confirms that `p/u/w` reverse states are the dominant
remaining memory source, but the Python-level local replay strategy loses the
speed benefit. With divergence `stride=2`, `state_cache_stride=20` lowers peak
memory to `0.569x` of production checkpoint, but total speed is only `0.946x`;
`state_cache_stride=2` is close to production memory (`1.063x`) but still only
`0.920x` total speed. Therefore do not continue stride enumeration in the
Python custom-autograd prototype. The next effective acoustic task should
return to production PyTorch checkpoint profiling and look for kernel-level
invariant hoisting or replay-cost reduction that preserves the default autograd
contract.

The first production checkpoint candidate round did not accept code changes.
Non-reentrant checkpoint failed on the tested NPU/TorchScript backward replay.
Source pre-scaling preserved exact outputs/loss/raw gradients but gave only
`1.025x` total speedup; empty receiver allocation preserved exact parity but
gave only `1.011x`. The earlier forward-wavefield placeholder candidate also
stayed below threshold. Do not continue small-allocation or per-step scalar
candidates. The next acoustic task must target measured stencil-update cost
directly, starting with a focused p/u/w timestep microbenchmark under the
production checkpoint shape.
```
