# Custom Autograd Forward Prototype

## Focus

This round starts the actual custom-operator route instead of adding more test
infrastructure.

## What Changed

`ADFWI/propagator/acoustic_operator.py` now supports:

```python
acoustic_pressure_operator(config, inputs, backend="custom_autograd_forward")
```

This backend wraps the pressure receiver forward path in a
`torch.autograd.Function` shell. It deliberately implements only forward
pressure output.

## Result

The prototype is useful only if it passes exact forward parity against the
operator reference backend:

```text
torch_reference
  |
  +-- production pressure-only forward_kernel

custom_autograd_forward
  |
  +-- same forward formula through custom-autograd shell
  +-- backward intentionally not implemented
```

This step does not claim speedup. It creates the minimum custom-autograd object
that can later receive an explicit backward/rematerialization implementation.

## Boundary

In scope:

- receiver pressure forward output;
- exact parity with `backend="torch_reference"`;
- explicit failure if `.backward()` is called.

Out of scope:

- no `vp.grad` support yet;
- no production `AcousticPropagator` integration;
- no full FWI loop testing for this backend.

## Next Step

Implement the backward side or stop the custom-autograd path. The next
meaningful acceptance gate is finite `vp.grad` parity against the reference
path on a tiny synthetic-energy case.
