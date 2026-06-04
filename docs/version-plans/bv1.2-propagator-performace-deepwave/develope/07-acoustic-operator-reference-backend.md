# Acoustic Operator Torch Reference Backend

## Focus

This round gives the acoustic operator boundary a real forward-pressure
reference backend while keeping production behavior unchanged.

## What Changed

`ADFWI/propagator/acoustic_operator.py` now supports:

```python
acoustic_pressure_operator(config, inputs, backend="torch_reference")
```

The reference backend calls the current production `forward_kernel` with:

- `pressure_only=True`;
- `save_forward_wavefield=False`;
- `checkpoint_segments=config.checkpoint_segments`.

The default backend remains `backend="compiled"`, which still raises
`CompiledAcousticOperatorUnavailable` until a compiled implementation exists.

## Result

This step does not claim speedup. Its purpose is to make the future compiled
backend testable against a stable reference through the same public operator
contract:

```text
operator contract
  |
  +-- torch_reference -> current production pressure path
  |
  +-- compiled        -> future low-level autograd implementation
```

## Boundary

In scope:

- establish a reference backend for receiver pressure parity;
- preserve exact production pressure output;
- keep the operator path separate from `AcousticPropagator.forward`.

Out of scope:

- no default-path production integration;
- no compiled kernel yet;
- no backward/gradient implementation in this step.

## Next Step

Start the compiled/autograd implementation behind `backend="compiled"` with the
same forward-pressure contract. The first acceptance gate is tiny forward
receiver pressure parity against `backend="torch_reference"`.
