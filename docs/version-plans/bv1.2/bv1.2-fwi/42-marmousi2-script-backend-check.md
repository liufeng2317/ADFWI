# Marmousi2 Script Backend Check

## Purpose

The minimal acoustic and elastic examples are intentionally tiny. This pass adds a bridge toward a real ADFWI case by introducing `scripts/examples/marmousi2_acoustic_backend_check.py`, a read-only script for the existing acoustic Marmousi2 notebook data.

The script is designed to verify that the real case can be loaded and reconstructed through the bv1.2 backend API without rewriting notebook outputs or historical example artifacts.

## What The Script Does

By default, the script:

- configures the backend once with `ADFWI.set_backend(...)`;
- reads `data/model/init_model.npz` or `data/model/true_model.npz`;
- reads `data/waveform/obs_data.npz`;
- reconstructs `Source`, `Receiver`, `Survey`, `SeismicData`, `AcousticModel`, and `AcousticPropagator`;
- checks shapes, dtype, device placement, finite values, and nonzero observed pressure norm;
- prints one JSON summary;
- writes no notebooks, figures, wavefields, inversion outputs, or data files.

The optional `--run-forward` flag runs one selected shot through the propagator. That mode is intentionally opt-in because the real Marmousi2 geometry is much heavier than the minimal examples.

## Commands

```bash
python -m py_compile scripts/examples/marmousi2_acoustic_backend_check.py
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device cpu
conda run -n adfwi python scripts/examples/marmousi2_acoustic_backend_check.py --device npu:0
```

## Validation Results

Local CPU and NPU read-only checks passed. Both runs reported:

- model grid: `nx=200`, `nz=88`, `dx=40.0`, `dz=40.0`, `nabc=30`;
- survey: `shots=40`, `receivers=200`, `nt=3000`, `dt=0.003`;
- observed pressure shape: `[40, 3000, 200]`;
- observed pressure norm: `4.604166507720947`;
- propagator damping shape: `[148, 260]`;
- finite `vp`, `rho`, observed `p/u/w`, and damping arrays.

The NPU run additionally confirmed that `vp`, `rho`, propagator device, and damping tensor were on `npu:0`.

## Next Step

The next optimization can build on this read-only case check in either of two ways:

1. add an opt-in Marmousi2 one-shot forward comparison script/report;
2. create a short Marmousi2 inversion smoke using a reduced shot/time subset and no persistent outputs.

Both should remain separate from the original notebooks so notebook figures and historical outputs stay fixed.
