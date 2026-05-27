# 67 - Acoustic Benchmark Scaffold

## Optimization Path

Continue from step 66 by adding the first benchmark scaffold before larger
performance-oriented changes. The goal is to make future propagator,
checkpoint, gradient-processing, and backend work measurable before changing
numerical kernels or FWI update rules.

## Change

- Added scripts/benchmark/acoustic_backend_benchmark.py.
- The benchmark runs a tiny in-memory acoustic forward/backward workload on a
  requested backend.
- The JSON report records command, seed, backend diagnostics, environment
  metadata, git commit/dirty status, model/survey dimensions, per-run timings,
  loss, pressure norm, gradient norm, and summary statistics.
- Added a CPU CLI smoke test that validates the report structure and confirms a
  finite nonzero vp gradient.

## Scientific Contract

- No propagator formula, misfit formula, FWI optimizer step, transform order, or
  model update logic changed.
- This step only adds measurement tooling around the existing acoustic forward
  and backward path.
- Because no FWI core behavior changed, numerical precision comparison against a
  baseline FWI run is not required for this step. The benchmark records loss,
  pressure norm, and vp gradient norm so future optimization steps can compare
  numerical drift explicitly.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python -m unittest tests/test_acoustic_benchmark.py: 1 test passed.
- conda run -n adfwi python -m py_compile scripts/benchmark/acoustic_backend_benchmark.py: passed.
- conda run -n adfwi python scripts/benchmark/acoustic_backend_benchmark.py --device cpu --warmup 0 --repeat 1 --nx 8 --nz 6 --nabc 2 --nt 8: printed a JSON report with status ok, CPU backend diagnostics, forward/backward timing, loss, pressure norm, and nonzero vp gradient norm.

The manual report showed git dirty status because it was run before committing
this step; this is expected and verifies that the benchmark captures worktree
state.

## Next Optimization Direction

Use this benchmark scaffold to collect baseline timings for checkpoint_segments
settings on CPU and the target NPU. After that, optimize the gradient processor
CPU/NumPy round-trip by adding an optional torch-native gradient-processing path
with explicit gradient-drift checks.
