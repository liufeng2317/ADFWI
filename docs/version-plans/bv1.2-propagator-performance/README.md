# bv1.2 Propagator Performance

This folder records the performance optimization plan for `ADFWI/propagator`.
It starts from the stable bv1.2 full-record Marmousi2 acoustic baseline and
keeps propagator changes bounded by explicit timing and numerical comparison.

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

First profiling record:

- `01-profile-first-bottleneck-probe.md`
