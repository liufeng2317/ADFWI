# Custom Autograd Remat Gradient Prototype

## Focus

This round implements the first backward-capable custom-autograd acoustic
pressure prototype.

## What Changed

`ADFWI/propagator/acoustic_operator.py` now supports:

```python
acoustic_pressure_operator(config, inputs, backend="custom_autograd_remat")
```

The backend:

- runs pressure receiver forward without saving full time history;
- saves only input tensors and scalar configuration;
- recomputes the pressure forward path during `backward`;
- calls `torch.autograd.grad` on the recomputed reference graph.

## Result

Tiny synthetic-energy parity passed:

| Quantity | Result |
| --- | --- |
| `vp.grad` finite | yes |
| `vp.grad` max abs diff vs `torch_reference` | `0.0` |

This is still a Python/Torch rematerialization prototype. It is not yet a
compiled operator and should not be treated as a production speedup.

## Boundary

In scope:

- pressure receiver output;
- rematerialized backward for tiny gradient parity;
- no default production-path integration.

Out of scope:

- no full Marmousi2 FWI loop yet;
- no compiled C++/CUDA/NPU backend yet;
- no claim about memory or speed before timed comparison.

## Next Step

Run a small timed comparison:

```text
torch_reference vs custom_autograd_remat
tiny/reduced synthetic-energy loss
forward + backward wall time
peak memory where available
vp.grad parity
```

If the remat backend is slower without meaningful memory reduction, stop this
Python remat route and move to compiled backend implementation.
