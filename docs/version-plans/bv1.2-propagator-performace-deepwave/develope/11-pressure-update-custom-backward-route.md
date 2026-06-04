# Pressure Update Custom Backward Route

## Focus

This round stops the Python remat direction and tests the more meaningful
Deepwave-inspired route: reduce autograd overhead for the hot pressure stencil
with a custom backward.

## Why This Is Different From Remat

Remat re-ran the same Python/Torch forward graph in backward. It preserved
parity but was slower and incompatible with production checkpointing.

The pressure-update custom backward route changes the autograd boundary:

```text
production pressure update
  |
  +-- many slice / copy-slice autograd nodes

custom pressure update
  |
  +-- one custom autograd node
  +-- explicit pressure-update backward formula
```

This is closer to the useful part of Deepwave: fewer PyTorch graph nodes around
the time-stepping stencil.

## Tests

Commands:

```bash
conda run -n adfwi python scripts/benchmark/acoustic_custom_pressure_update_probe.py \
  --device npu:0 --nx 128 --nz 64 --shots 8 --repeat 5 --warmup 2 \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_custom_pressure_update_probe_20260604.json

conda run -n adfwi python scripts/benchmark/acoustic_custom_pressure_update_probe.py \
  --device npu:0 --nx 128 --nz 64 --shots 40 --repeat 5 --warmup 2 \
  --output docs/version-plans/bv1.2-propagator-performace-deepwave/develope/acoustic_custom_pressure_update_probe_40shot_20260604.json
```

## Results

| Case | Total speedup mean | Backward speedup mean | Forward speedup mean | Numerical parity |
| --- | ---: | ---: | ---: | --- |
| `128x64`, 8 shots | `1.292x` | `1.344x` | `1.155x` | output exact; grad max abs `1.42e-14` |
| `128x64`, 40 shots | `1.314x` | `1.385x` | `1.165x` | output exact; grad max abs `3.55e-15` |

The 40-shot case has more timing variance, including one slower run, but the
mean still supports this direction.

## Interpretation

This is the first current-branch result that looks like a meaningful operator
optimization:

- it changes the autograd boundary, not only storage policy;
- it improves the local hot stencil in both forward and backward;
- it preserves output exactly and keeps gradient differences near float32 noise.

## Integration Boundary

Do not replace the default scripted production kernel directly.

Current `step_forward` and `step_forward_pressure_only` are TorchScript
functions. A Python `torch.autograd.Function` cannot be inserted into those
scripted functions cleanly.

The next integration step should therefore be opt-in:

```text
new non-scripted pressure-only segment
  |
  +-- uses custom pressure update
  +-- keeps source injection, velocity updates, receiver sampling unchanged
  +-- compares against production pressure-only forward/gradient
```

Only if the opt-in segment gives a real FWI-loop speedup should it be considered
for production integration or replacement by a compiled backend.

## Next Step

Build a minimal opt-in pressure-only segment using the custom pressure-update
autograd function. Validate:

1. tiny forward pressure parity;
2. tiny synthetic-energy `vp.grad` parity;
3. reduced `3 shot x 3 iter` timing before any full-case run.
