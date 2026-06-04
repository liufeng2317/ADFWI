# Deepwave-Inspired Implementation Change Log

This is the compact active record for this branch. It should stay short.

| Date | Step | Result | Next action |
| --- | --- | --- | --- |
| 2026-06-04 | Created `bv1.2-deepwave-propagator` branch from `bv1.2@4507a08` | dedicated branch for Deepwave-inspired propagator design and implementation | define acoustic operator contract before adding new kernel code |
| 2026-06-04 | Added acoustic operator contract document | clarified ADFWI formula boundary, first pressure-only target, storage-policy boundary, and validation order | implement a separate forward-pressure prototype and compare against production receiver output |
| 2026-06-04 | Built production acoustic FWI baseline matrix | measured 1/3/40 shots with 3/10 FWI iterations on full-record Marmousi2, `checkpoint_segments=10`, `npu:0`; all runs finite | use this as the pre-implementation comparison table for custom operator/storage-policy work |
| 2026-06-04 | Added production checkpoint sweep | measured `checkpoint_segments=1/5/10`; checkpoint=1 is faster but much higher memory, checkpoint=5 is slower and higher memory than checkpoint=10 | target ckpt=1-like speed without accepting its memory increase |
