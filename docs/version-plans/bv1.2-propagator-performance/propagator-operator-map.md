# Propagator Operator Map

This map describes where computation happens and what each file owns. It is
intended to prevent performance work from drifting into unrelated cleanup.

## Public Entry Points

```text
ADFWI.propagator.AcousticPropagator
ADFWI.propagator.ElasticPropagator
ADFWI.propagator.GradProcessor
ADFWI.propagator.TorchGradProcessor
```

The public wrappers should stay thin. They prepare model/survey/backend state
and delegate numerical propagation to the kernel modules.

## Acoustic Operator

File: `ADFWI/propagator/acoustic_kernels.py`

Main functions:

- `pad_torchSingle`: pads velocity and density fields for absorbing boundaries.
- `step_forward`: runs a checkpoint segment over time.
- `forward_kernel`: prepares padded tensors, initializes state/output buffers,
  segments the source wavelet, calls `checkpoint(step_forward)`, and returns the
  waveform dictionary.

Output contract:

```text
p, u, w
forward_wavefield_p, forward_wavefield_u, forward_wavefield_w
```

Observed performance-sensitive operations:

- full timestep loop in `step_forward`;
- source injection with per-step index use;
- receiver sampling for all three components;
- forward wavefield accumulation for all three components at every timestep;
- temporary receiver and wavefield buffers allocated inside each checkpoint
  segment;
- checkpoint call even when `checkpoint_segments=1`.

First optimization target: acoustic measurement and low-risk invariant cleanup.

## Elastic Operator

File: `ADFWI/propagator/elastic_kernels.py`

Main functions:

- `DiffCoef`: finite-difference coefficients for each order.
- `Dxfm/Dzfm/Dxbm/Dzbm` variants: finite-difference derivative operators for
  orders 4, 6, 8, and 10.
- `pad_torchSingle`: pads elastic parameters and boundary arrays.
- `step_forward_PML_*order`: PML timestep segment for order 4/6/8/10.
- `step_forward_ABL_*order`: ABL timestep segment for order 4/6/8/10.
- `forward_kernel`: prepares tensors, chooses PML/ABL and finite-difference
  order, calls checkpointed step functions, and returns the waveform dictionary.

Output contract:

```text
txx, tzz, txz, vx, vz
forward_wavefield_txx, forward_wavefield_tzz, forward_wavefield_txz,
forward_wavefield_vx, forward_wavefield_vz
```

Observed performance-sensitive operations:

- eight step functions with duplicated structure;
- derivative operators that use advanced indexing ranges;
- five receiver components recorded per timestep;
- five forward wavefield accumulations per timestep;
- checkpointed segment calls for PML and ABL branches;
- padding of all elastic moduli and boundary fields at each forward call.

First elastic target: profiling and parity tests only, after acoustic workflow
is stable.

## Boundary Setup

File: `ADFWI/propagator/boundary_condition.py`

Boundary functions are NumPy setup functions:

- `bc_pml`
- `bc_sincos`
- `bc_gerjan`
- `bc_pml_xz`

These are not the main timestep hot path. Optimize them only if profiling shows
construction dominates a small repeated-forward workflow.

## Gradient Post-Processing

File: `ADFWI/propagator/gradient_process.py`

Components:

- `GradProcessor`: NumPy/SciPy path.
- `TorchGradProcessor`: tensor path intended to avoid CPU round trips.
- `smooth2d`: helper used by legacy workflows.

This is part of inversion runtime performance, not wave propagation itself.
Treat it as a separate phase after propagator forward timing is stable.

## Experimental Boundary-Saving Kernel

File: `ADFWI/propagator/acoustic_kernels_bs.py`

This file implements an experimental boundary-saving checkpoint approach and
sets autograd anomaly detection. It is not imported by the public acoustic
wrapper. Keep it outside the default performance route until a dedicated
experiment proves:

- forward waveform parity;
- gradient parity;
- memory reduction;
- wall-time impact;
- no global autograd side effects in normal imports.

