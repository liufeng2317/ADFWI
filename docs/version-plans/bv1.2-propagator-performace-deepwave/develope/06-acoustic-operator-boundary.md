# Acoustic Operator Boundary Implementation

## Focus

This round moves the branch from repeated Python prototype testing toward a
clear acoustic operator implementation boundary.

The production acoustic kernel is not changed in this step.

## What Changed

Added `ADFWI/propagator/acoustic_operator.py` with:

- `AcousticOperatorConfig`: scalar metadata for a pressure-only acoustic
  operator;
- `AcousticOperatorInputs`: tensor input contract for source, receiver, model,
  damping, and wavelet tensors;
- storage-policy names: `device`, `checkpoint`, `none`;
- `acoustic_pressure_operator(...)`: explicit future dispatch point that
  validates inputs and raises `CompiledAcousticOperatorUnavailable` until a
  compiled backend exists.

Added `tests/test_acoustic_operator_contract.py` to lock the contract before
kernel implementation.

## Result

This is not a speedup claim. It is an implementation boundary:

```text
AcousticPropagator.forward
  |
  +-- remains current production path

acoustic_operator.py
  |
  +-- future compiled/autograd pressure-only operator contract
  +-- no silent fallback
  +-- no production behavior change
```

## Boundary

In scope:

- define shapes, scalar metadata, and storage-policy names;
- fail clearly when a compiled backend is requested but not available;
- keep current production numerical behavior unchanged.

Out of scope:

- no custom C++/CUDA/NPU kernel yet;
- no default `AcousticPropagator` integration yet;
- no performance claim from this step.

## Next Step

Implement the first real compiled/autograd prototype behind this boundary:

1. pressure receiver output only;
2. no wavefield summaries;
3. compare against production on tiny forward parity before backward work;
4. keep the prototype opt-in until forward and gradient gates pass.
