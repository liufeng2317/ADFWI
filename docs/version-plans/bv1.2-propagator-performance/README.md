# bv1.2 Propagator Performance

This folder is the concise entry point for acoustic propagator performance work.

Read in this order:

1. `propagator-performance-summary.md`
2. `propagator-performance-master-plan.md`
3. `performance-test-matrix.md`

Detailed round-by-round notes and raw JSON outputs are archived under:

- `archive/round-logs/`
- `archive/results/`

Do not use archived round logs as the active optimization queue. They are kept
only for traceability.

## Current Decision

Continue performance work only from the accepted summary and test matrix.

The next valid optimization must:

- target the production acoustic hot path,
- preserve forward waveform, loss, and `vp` gradient parity,
- report runtime before/after on a named case,
- stop after one bounded change.

Ascend custom-op work is paused until the multi-block custom-op copy contract is
resolved outside production ADFWI.

