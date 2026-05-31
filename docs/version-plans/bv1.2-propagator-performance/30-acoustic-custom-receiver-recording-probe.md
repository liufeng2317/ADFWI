# Acoustic Custom Receiver Recording Probe

Date: 2026-05-31

## Purpose

This probe extends the isolated acoustic custom-gradient recurrence prototype
with receiver recording. It is still not a production-kernel change.

The goal is to check whether the custom-gradient route remains numerically
stable after adding the main forward outputs used by acoustic inversion:

- final pressure and particle-velocity states;
- per-step receiver records for `p`, `u`, and `w`;
- source injection;
- free-surface boundary writes.

## Change

Updated:

- `scripts/benchmark/acoustic_custom_multistep_update_probe.py`

The benchmark now supports:

- `--receiver-recording`
- `--receivers`
- `--receiver-depth`

When receiver recording is enabled, both the reference PyTorch recurrence and
the custom-gradient recurrence record `p/u/w` at the same receiver indices at
every timestep. These recorded traces are included in the loss and in the
output comparison.

## Command

```bash
timeout 480s conda run -n adfwi python scripts/benchmark/acoustic_custom_multistep_update_probe.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 1 \
  --repeat 3 \
  --steps 20 \
  --shots 1 \
  --nx 100 \
  --nz 50 \
  --nabc 20 \
  --pressure-scale 1e-3 \
  --velocity-scale 1e-6 \
  --kappa-scale 1e-3 \
  --alpha1-scale 1e-3 \
  --alpha2-scale 1e-3 \
  --source-injection \
  --free-surface-boundary-write \
  --receiver-recording \
  --receivers 16 \
  --receiver-depth 1 \
  --source-depth 1 \
  --source-scale 1e-4 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_custom_multistep_receiver_probe_20260531.json
```

## Result

Reference record:

- `acoustic_custom_multistep_receiver_probe_20260531.json`

| Metric | Value |
| --- | --- |
| Device / dtype | `npu:0`, `float32` |
| Steps | `20` |
| Model | `shots=1`, `nx=100`, `nz=50`, `nabc=20` |
| Receivers | `16` |
| Source injection | enabled |
| Free-surface boundary write | enabled |
| Receiver recording | enabled |
| Loss absolute difference | `0.0` |
| Output maximum absolute difference | `0.0` |
| Output maximum relative difference | `0.0` |
| Gradient maximum absolute difference | `4.815348120246199e-11` |
| Gradient maximum relative difference | `0.010578576475381851` |
| Forward mean speedup | `0.891226352577719x` |
| Backward mean speedup | `1.978189721514546x` |
| Total mean speedup | `1.544977718813799x` |

All final-state outputs and recorded receiver outputs matched exactly in this
isolated probe. The largest gradient absolute difference was on `alpha2`.
The relative difference is larger because the corresponding gradient norm is
small; this is acceptable for an isolated feasibility probe, but it is not yet
enough for production integration.

## Decision

The custom-gradient acoustic route remains viable after adding receiver
recording. The backward speedup signal is still strong, while the prototype
forward is slower because it runs through a Python `autograd.Function` wrapper
and intentionally keeps the implementation simple for parity testing.

Do not integrate this into the production acoustic kernel yet.

## Next Direction

The next task should stay on the same main line:

```text
Build an opt-in experimental acoustic-kernel parity harness that compares a
tiny production `forward_kernel` run against the custom-gradient recurrence for
receiver records, loss, and raw `vp.grad`.
```

That is the required gate before considering any production-facing
implementation. It keeps the work aligned with the measured bottleneck:
differentiable recurrent timestep backward overhead.
