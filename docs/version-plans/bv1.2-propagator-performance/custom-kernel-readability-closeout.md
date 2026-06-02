# Custom Kernel Readability Closeout

## Scope

This pass only cleans up names and stale code around the acoustic performance
paths. It does not change the finite-difference equations, receiver sampling,
autograd formulas, checkpoint policy, or public propagator API.

## Changes

- Renamed internal helpers in `ADFWI/propagator/acoustic_custom_kernels.py` so
  their names describe the actual role:
  - forward step with saved divergence terms
  - pressure/velocity divergence recomputation
  - backward step from saved divergence terms
  - saved-state and rematerialized chunk autograd functions
- Kept public entry points unchanged:
  - `custom_chunk_forward_kernel`
  - `rematerialized_custom_chunk_forward_kernel`
- Removed stale debug remnants from `ADFWI/propagator/acoustic_kernels.py`:
  - unused `numpy` import
  - unused `wavefields` list
  - commented absolute-path wavefield dump block

## Remaining Boundary

`ADFWI/propagator/acoustic_kernels_bs.py` remains a research prototype file. It
is not wired into the default propagator or the opt-in custom chunk path. If it
is no longer needed, archive or delete it in a separate explicit cleanup pass.
