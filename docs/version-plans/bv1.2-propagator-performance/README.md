# bv1.2 Propagator Performance

This folder records the performance optimization plan for `ADFWI/propagator`.
It starts from the stable bv1.2 full-record Marmousi2 acoustic baseline and
keeps propagator changes bounded by explicit timing and numerical comparison.

Start here:

- `propagator-performance-master-plan.md`: execution rules, phase order,
  baselines, stop criteria, and the next required profiling task.
- `ad-fwi-efficiency-method-map.md`: method map for AD-FWI performance work,
  including what is low-risk, opt-in, or research-only.
- `performance-test-matrix.md`: validation commands and acceptance rules.
- `propagator-operator-map.md`: operator ownership and hot-path map.

Current scope:

- measure the current acoustic propagator bottleneck before kernel edits;
- analyze the acoustic and elastic propagator operator paths;
- define the first performance optimization route from measured bottlenecks;
- define the test matrix before changing kernel behavior.

Current baseline:

- baseline record: `docs/version-plans/bv1.2/full-record-marmousi2-baseline.md`
- branch for future work: `bv1.2-propagator-performance`
- forward anchor: Marmousi2 acoustic full-record, `checkpoint_segments=1`
- inversion anchor: Marmousi2 acoustic full-record, 300 iterations

Do not treat this folder as a task queue. Each future performance round should
start with a measured bottleneck, select one bounded target from the route, run
the matching tests, record the before/after result, then stop.

Current execution rule:

```text
No further kernel optimization until Phase A produces an end-to-end cost
breakdown for one reduced differentiable acoustic FWI iteration.
```

First profiling record:

- `01-profile-first-bottleneck-probe.md`
- `09-acoustic-fwi-iteration-profile.md`

Current measured next direction:

```text
Reduced acoustic FWI iteration is dominated by backward cost
(`17.78 s`, `61.01%`). Continue with Phase B: acoustic AD graph and backward
cost diagnostics.
```

Phase B policy diagnostic:

- `10-acoustic-fwi-wavefield-policy-profile.md`
- `11-acoustic-fwi-wavefield-policy-repeat.md`
- Under `GradProcessor(forw_illumination=False)`, skipping acoustic
  forward-wavefield summaries preserved loss, raw gradients, processed
  gradients, and updated `vp` exactly in the reduced single-iteration
  comparison.
- Repeated paired runs kept all numerical differences at `0.0`; total speedup
  ranged from `1.06x` to `1.26x`, with candidate runtime stable around `22 s`.
  This is accepted as an opt-in path and should not become the default.

Current default-path next direction:

```text
Return to Phase B and profile the default acoustic backward path at finer
granularity before making another kernel change.
```

Default backward operator profile:

- `12-acoustic-default-backward-operator-profile.md`
- `13-acoustic-timestep-update-microbenchmark.md`
- `14-acoustic-checkpoint-segment-sweep.md`
- `15-acoustic-compile-feasibility.md`
- `16-acoustic-receiver-recording-stack-probe.md`
- `17-acoustic-pressure-inner-state-probe.md`
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
- Representative NPU profile shows default backward dominated by sliced update
  autograd overhead: `SliceBackward0`, `copy_`, `zero_`, `zeros`, and
  `empty_tensor`.
- The first bounded update-style microbenchmark rejected `torch.cat`
  functional reconstruction: it preserved output and gradients exactly, but was
  slower than the current sliced-assignment style in stable NPU timings.
- Checkpoint segment sweep shows `checkpoint_segments=1` remains fastest when
  memory is sufficient; segmenting helps memory control but increases backward
  recomputation cost and total time for the measured NPU case.
- `torch.compile` is not a current production route on this NPU environment:
  the default inductor path fails before the first compiled run because
  `triton` is unavailable. The `backend=eager` control preserves output and
  gradients exactly, but provides no useful optimizing backend.
- Receiver-output list stacking was rejected for the production acoustic
  kernel: isolated output recording was faster with exact gradient parity, but
  the reduced Marmousi2 FWI iteration was slower (`25.68 s -> 26.75 s`). The
  production kernel change was reverted and only the benchmark record remains.
- A pressure inner-state recurrence rewrite was rejected as a small default
  optimization: the isolated Python reference hit autograd in-place version
  constraints before timing comparison, showing that this route is a larger
  research rewrite rather than a safe kernel edit.
- Acoustic default-path Phase B should pause.
- Acoustic Phase D found a useful opt-in path: `TorchGradProcessor` reduced
  gradient-processing time in the reduced Marmousi2 profile from `0.644 s` to
  `0.052 s`, with total single-iteration speedup `1.071x` and stable reduced
  FWI numerical metrics.
- The 10-iteration reduced Marmousi2 comparison stayed stable: final loss and
  `vp_update_norm` matched exactly, max loss relative difference was `8.54e-08`,
  and total compute improved by `1.072x`. Continue by exposing
  `gradient_processor="torch"` as an explicit acoustic NPU performance option,
  not as a global default.
- Acoustic reduced/full-record validation scripts now expose
  `--gradient-processor legacy|torch`; default remains `legacy`. A reduced
  1-iteration NPU smoke confirmed the `torch` option path and summary output.
- Full-record 10-iteration validation did not show an end-to-end win for the
  torch-native gradient processor: numerical differences stayed negligible, but
  wall time was slower (`232.00 s -> 240.73 s`). Keep `legacy` as the default
  and treat `torch` as an explicit reduced/NPU profiling option.
- Full-record single-iteration profiling shows the current acoustic bottleneck
  is still differentiable propagation: backward `16.93 s` (`59.86%`) and
  forward `9.83 s` (`34.78%`) of a `28.27 s` measured iteration.
- Full-record backward operator profiling confirms the earlier small-profile
  conclusion: backward is dominated by slice/copy/zero/allocation autograd
  overhead, not a single physical math operator. Stop repeating this profile
  and move to a timestep update feasibility decision.
- Acoustic timestep state-update rewrite is not accepted as a small
  optimization task. Remaining meaningful changes would alter the autograd
  representation of the recurrent wave-equation update and should move to a
  separate custom-autograd/adjoint research route.
- This branch remains the active performance branch. The next high-impact path
  is a custom-gradient/adjoint-style acoustic update prototype with strict
  output and gradient parity gates, starting from a tiny pressure-update probe.
- The first custom-autograd pressure-update probe produced a positive signal:
  output/loss matched exactly, maximum gradient absolute difference was
  `2.27e-13`, and backward speedup averaged `1.67x` on the isolated update.
- The complete one-step `p/u/w` custom timestep prototype also produced a
  positive signal: outputs/loss matched exactly, maximum gradient absolute
  difference was `1.36e-12`, and backward speedup averaged `1.87x`.
- The 20-step recurrence prototype preserved final outputs and loss exactly,
  kept gradient maximum absolute difference at `5.68e-14`, and kept backward
  speedup around `1.87x`.
