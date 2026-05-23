# Misfit Function Support Matrix

This document summarizes the current backend support status of misfit functions in `ADFWI/fwi/misfit` for bv1.2.

Status labels:

- `Verified`: covered by tensor-level smoke tests on the listed device.
- `Likely`: implementation is tensor-local and expected to work, but not yet included in the default smoke group.
- `CPU only`: works or is expected to work on CPU, but not currently supported on NPU.
- `Experimental`: implementation depends on third-party or CPU/NumPy/SciPy paths and needs additional validation.
- `Not exported`: not currently exported from `ADFWI.fwi.misfit.__init__`.

## Quick Support List

| Misfit class | Short name | Exported | Main idea | CPU | NPU | Notes |
|---|---|---:|---|---|---|---|
| `Misfit_waveform_L1` | `L1` | Yes | L1 waveform difference | Verified | Verified | Pure torch. |
| `Misfit_waveform_L2` | `L2` | Yes | L2 waveform difference | Verified | Verified | Pure torch; default ADFWI baseline. |
| `Misfit_waveform_smoothL1` | `SmoothL1` | Yes | PyTorch SmoothL1 waveform loss | Verified | Verified | Uses `nn.SmoothL1Loss`. |
| `Misfit_waveform_studentT` | `StudentT` | Yes | Robust Student-t waveform loss | Verified | Verified | Pure torch. |
| `Misfit_weighted_L1_and_L2` | `WeightedL1L2` | Yes | Iteration-weighted L1/L2 composite | Verified | Verified | Wraps L1 and L2. |
| `Misfit_global_correlation` | `GC` | Yes | Global correlation loss | Verified | Verified | Tensor-local loops; dtype/device cleaned in bv1.2. |
| `Misfit_traveltime` | `TravelTime` | Yes | Differentiable soft travel-time shift | Verified | Verified | Tensor-level smoke passed; still experimental physically. |
| `Misfit_NIM` | `NIM` | Yes | Normalized Integration Method | Verified | Verified | Custom autograd; tensor-level CPU/NPU smoke passed. |
| `Misfit_envelope` | `Envelope` | Yes | Envelope or instantaneous phase misfit | Verified | CPU only | NPU currently fails on `complex64` `torch.abs` after FFT/Hilbert transform. |
| `Misfit_weighted_ECI` | `WECI` | Yes | Weighted envelope + global correlation | Verified | CPU only | Depends on `Envelope`; same NPU complex tensor limitation. |
| `Misfit_wasserstein_sinkhorn` | `Sinkhorn` | Yes | GeomLoss Sinkhorn divergence | Experimental | Experimental | Time-coordinate dtype/device cleaned, but GeomLoss NPU support is not validated. |
| `Misfit_sdtw` | `SoftDTW` | Yes | Soft-DTW divergence through `pysdtw` | Experimental | CPU/CUDA only pending fix | Current implementation has CUDA-specific logic; NPU support is not validated. |
| `Misfit_weighted_DTW_GC` | `WDGC` | Yes | Weighted SoftDTW + global correlation | Experimental | CPU/CUDA only pending fix | Depends on `SoftDTW`; NPU support blocked by SoftDTW. |
| `Misfit_Wasserstein1` | `Wasserstein1` | No | Torch cumulative Wasserstein-1 style loss | Likely | Likely | Not exported; dtype/device cleaned but not in smoke default group. |
| `Misfit_wasserstein_1d` | `Wasserstein1D` | No | POT/GeomLoss 1D Wasserstein | Experimental | Experimental | Uses third-party optimal transport; not validated on NPU. |
| `gsot` | `GSOT` | No | Generalized sorting optimal transport helper | CPU only | CPU only | Uses SciPy `linear_sum_assignment` and NumPy loops; not AD/backend friendly. |

## Current Verified Smoke Commands

Default CPU/NPU verified group:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0
```

Default group:

```text
L1,L2,SmoothL1,StudentT,WeightedL1L2,GC,TravelTime,NIM
```

Envelope/WECI CPU-only check:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu --misfits Envelope,WECI
```


## Acoustic Mini Inversion Smoke

The same portable group is also covered by one-step `AcousticFWI` mini inversion smoke through:

```bash
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2
```

Supported `--misfit` values for this mini inversion smoke are:

```text
L1, L2, SmoothL1, StudentT, WeightedL1L2, GC, TravelTime, NIM
```

Validation on the local CPU/NPU environment passed for all listed values. `SmoothL1` and `StudentT` use a larger script default learning rate because their gradients are very small in the tiny synthetic smoke model; users can still override this with `--lr`.

## Recommended Usage for bv1.2

For CPU/NPU portable acoustic inversion, prefer:

```text
L1, L2, SmoothL1, StudentT, WeightedL1L2, GC, TravelTime, NIM
```

Use with caution:

```text
Envelope, WECI
```

These are currently CPU-only in the local NPU environment because the Hilbert/FFT implementation produces complex tensors that the NPU runtime cannot fully process.

Treat as experimental until separately validated:

```text
SoftDTW, WDGC, Wasserstein_sinkhorn, Wasserstein_1d, Wasserstein1
```

## Notes for Future Refactoring

- Fix `SoftDTW` backend detection before testing NPU. It should distinguish `cpu`, `cuda`, and `npu` explicitly instead of treating all non-CPU devices as CUDA.
- Keep `Envelope`/`WECI` marked CPU-only until an NPU-compatible envelope implementation is added.
- Validate `GeomLoss` and POT-based Wasserstein losses independently before advertising NPU support.
- Keep this file synchronized with `docs/version-plans/bv1.2-misfit-backend-audit.md`.
