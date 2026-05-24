# bv1.2 Misfit Backend Audit and Refactor Plan

This document audits `ADFWI/fwi/misfit` for CPU/NPU/backend compatibility before changing implementation code. The goal is to extend bv1.2 backend validation from propagators and FWI smoke tests to commonly used misfit functions.

## Scope

Covered files:

- `L1.py`, `L2.py`, `SmoothL1.py`
- `Weighted_L1_L2.py`, `StudentT.py`, `GlobalCorrelation.py`
- `Envelope.py`, `Weci.py`
- `TravelTime.py`
- `Normalized_Integration_method.py`, `Wasserstein_1.py`
- `Wasserstein_sinkhorn.py`, `Wasserstein_1d.py`
- `SoftDTW.py`, `WDGC.py`

The first pass focuses on acoustic waveform shapes used by `AcousticFWI`: `[shot, time, receiver]`.

## Current Findings

| Misfit | Exported | Implementation type | Backend status | Risk level | Notes |
|---|---:|---|---|---|---|
| `Misfit_waveform_L1` | yes | Pure torch | Likely CPU/NPU safe | Low | Uses tensor-local arithmetic only. |
| `Misfit_waveform_L2` | yes | Pure torch | Likely CPU/NPU safe | Low | Uses masks and reductions on input device. |
| `Misfit_waveform_smoothL1` | yes | PyTorch module | Likely CPU/NPU safe | Low | `nn.SmoothL1Loss` follows input device/dtype. |
| `Misfit_waveform_studentT` | yes | Pure torch | Likely CPU/NPU safe | Low | Does not call `super().__init__()`, but no backend issue. |
| `Misfit_weighted_L1_and_L2` | yes | Composite pure torch | Likely CPU/NPU safe | Low | Wraps L1/L2. Python scalar weight is fine. |
| `Misfit_global_correlation` | yes | Pure torch loops | Mostly safe | Low-Medium | Creates `rsd` on `obs.device`, but default dtype follows torch default, not `obs.dtype`. Should use `dtype=obs.dtype`. |
| `Misfit_envelope` | yes | Torch FFT + scripted helpers | Needs smoke test | Medium | Uses `torch.fft` and `np.pi` constants inside scripted unwrap. Likely device-safe, but NPU FFT support must be verified. `rsd` should use `obs.dtype`. |
| `Misfit_weighted_ECI` | yes | Envelope + GC | Depends on Envelope | Medium | Safe if Envelope and GC pass. |
| `Misfit_traveltime` | yes | Torch conv1d loops | Needs small dtype fix + smoke | Medium | `tt_lags` hardcodes `torch.float32`; `rsd` does not set dtype. Also normalization may divide by zero for empty traces. |
| `Misfit_NIM` | yes | Custom autograd Function | Needs smoke test | Medium | Tensor-local operations. `torch.ones(...).to(device)` should preserve dtype explicitly. Custom backward should be validated with acoustic mini smoke. |
| `Misfit_Wasserstein1` | not currently exported | Pure torch W1-like implementation | Needs small dtype fix + optional export decision | Medium | Similar transform helpers to NIM. Uses asserts on tensor reductions; may behave awkwardly on accelerators. |
| `Misfit_wasserstein_sinkhorn` | yes | GeomLoss `SamplesLoss` | Dependency/backend uncertain | High | Builds time coordinates from NumPy then moves to device; should use `torch.arange(..., device=obs.device, dtype=obs.dtype)`. GeomLoss NPU support is unknown. |
| `Misfit_wasserstein_1d` | not exported | POT/GeomLoss/SciPy hybrid | CPU/third-party boundary | High | Uses `ot.lp.wasserstein_1d`; support on NPU is uncertain. `gsot` uses SciPy assignment and NumPy loops, so it is CPU-bound/non-AD-friendly. |
| `Misfit_sdtw` | yes | `pysdtw` | CUDA-specific logic, NPU unsafe | High | `use_cuda=device != "cpu"` is wrong for `torch.device`; for NPU this likely requests CUDA path. Needs explicit capability boundary or CPU fallback. |
| `Misfit_weighted_DTW_GC` | yes | SoftDTW + GC | NPU unsafe until SoftDTW fixed | High | Inherits SoftDTW limitation. |

## Cross-Cutting Issues

1. **Dtype inheritance**

   Several misfits allocate tensors with only `device=obs.device` or `.to(device)`:

   ```python
   torch.zeros(..., device=obs.device)
   torch.ones(f.shape).to(device)
   torch.tensor(...).to(device)
   ```

   These should generally become:

   ```python
   torch.zeros(..., device=obs.device, dtype=obs.dtype)
   torch.ones_like(f)
   torch.arange(..., device=obs.device, dtype=obs.dtype)
   ```

2. **CUDA-specific assumptions**

   `SoftDTW.py` currently uses:

   ```python
   use_cuda = device != "cpu"
   ```

   Here `device` is a `torch.device`, so this condition is true even for CPU (`torch.device("cpu") != "cpu"`) and also true for NPU. This should not be used as a backend capability test.

3. **Third-party accelerator support**

   `pysdtw`, `geomloss`, and `ot.lp.wasserstein_1d` should not be assumed to support NPU. These misfits need explicit capability checks or documentation before being advertised as CPU/NPU safe.

4. **Silent CPU/NumPy boundaries**

   Some code imports NumPy only for constants or ranges, which is easy to fix. Other code, especially `gsot`, uses SciPy/NumPy algorithms and should be marked CPU-bound unless rewritten.

5. **Trace masking and zero normalization**

   Several misfits normalize by max/norm/sum without consistent zero guards. This is separate from backend support but should be tested because NPU smoke tests can surface NaNs quickly.

## Proposed Refactor Plan

### Phase 1: Low-Risk Dtype/Device Cleanup

Target files:

- `GlobalCorrelation.py`
- `TravelTime.py`
- `Normalized_Integration_method.py`
- `Wasserstein_1.py`
- `Wasserstein_sinkhorn.py`

Planned changes:

- Replace `torch.zeros(..., device=device)` with `dtype=obs.dtype` or `dtype=syn.dtype` where appropriate.
- Replace `torch.ones(f.shape).to(device)` with `torch.ones_like(f)`.
- Replace `torch.tensor(np.arange(...)).to(device)` with `torch.arange(..., device=device, dtype=obs.dtype) * dt`.
- Replace `torch.float32` hardcoding in `TravelTime.calculate_time_shift` with `wave1.dtype`.

Expected risk: low, because behavior should remain mathematically equivalent while improving dtype/device consistency.

### Phase 2: Misfit Smoke Harness

Extend `scripts/smoke/acoustic_mini_inversion_smoke.py` or add a new focused script to test misfits on synthetic tensors first.

Recommended first script:

```text
scripts/smoke/misfit_backend_smoke.py
```

Suggested behavior:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu --misfits L1,L2,SmoothL1,GC,Envelope,TravelTime,NIM
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0 --misfits L1,L2,SmoothL1,GC,Envelope,TravelTime,NIM
```

Each test should:

- create small `obs` and `syn` tensors on the selected backend;
- call the misfit;
- call `loss.backward()`;
- verify finite scalar loss and finite nonzero gradient on `syn`;
- return a JSON summary.

Start with:

- `L1`
- `L2`
- `SmoothL1`
- `StudentT`
- `Weighted_L1_L2`
- `GlobalCorrelation`
- `TravelTime`
- `NIM`

Then add:

- `Envelope` if NPU FFT passes;
- `WECI` if Envelope + GC pass.

### Phase 3: Capability Boundaries for Heavy Misfits

Target files:

- `SoftDTW.py`
- `WDGC.py`
- `Wasserstein_sinkhorn.py`
- `Wasserstein_1d.py`

Planned decisions:

1. `SoftDTW`
   - Replace `device != "cpu"` with a proper check:

     ```python
     use_cuda = obs.device.type == "cuda"
     ```

   - For NPU, either raise a clear `RuntimeError` or route through a CPU-safe path only if gradients remain correct and documented.

2. `WDGC`
   - Mark NPU unsupported until `SoftDTW` backend behavior is explicit.

3. `Wasserstein_sinkhorn`
   - Fix time coordinate creation first.
   - Test `geomloss` on CPU and NPU. If NPU fails, raise a clear unsupported-backend message instead of failing deep inside the dependency.

4. `Wasserstein_1d` / `gsot`
   - Keep unexported for now.
   - Mark `gsot` as CPU-bound because it uses SciPy `linear_sum_assignment` and NumPy loops.

### Phase 4: Acoustic Mini Inversion Misfit Options

After tensor-level smoke passes, extend acoustic mini inversion smoke with:

```bash
--misfit L2
--misfit GC
--misfit Envelope
--misfit TravelTime
--misfit NIM
```

This should only include misfits that have already passed tensor-level CPU/NPU smoke.

### Phase 5: Documentation and Examples

Update backend docs with a support matrix:

- supported on CPU/NPU;
- CPU-only;
- CUDA-only;
- experimental / not validated.

Avoid changing notebooks until script-level validation is stable.

## Implementation Status

Implemented in the current `bv1.2` working tree:

- Phase 1 dtype/device cleanup for:
  - `GlobalCorrelation.py`;
  - `TravelTime.py`;
  - `Normalized_Integration_method.py`;
  - `Wasserstein_1.py`;
  - `Wasserstein_sinkhorn.py`.
- Added tensor-level smoke test script:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0
```

Default validated misfit group:

```text
L1,L2,SmoothL1,StudentT,WeightedL1L2,GC,TravelTime,NIM
```

Validation result on the local `adfwi` CPU+NPU environment:

- CPU: default group status `ok`; all losses finite and all synthetic-waveform gradients finite/nonzero.
- NPU `npu:0`: default group status `ok`; all losses finite and all synthetic-waveform gradients finite/nonzero.
- CPU: `Envelope,WECI` status `ok`.
- NPU `npu:0`: `Envelope,WECI` failed because the current NPU runtime does not support `torch.abs` on `complex64` tensors produced by the Hilbert/FFT path. Keep these CPU-only or redesign the envelope implementation for NPU before advertising NPU support.

## Acoustic Mini Inversion Follow-Up

Implemented `--misfit` support in `scripts/smoke/acoustic_mini_inversion_smoke.py` for the tensor-smoke-verified portable group:

```text
L1, L2, SmoothL1, StudentT, WeightedL1L2, GC, TravelTime, NIM
```

Validation result on the local `adfwi` CPU+NPU environment:

- CPU: all eight values completed one `AcousticFWI` iteration with finite loss, finite/nonzero `vp` gradient, and finite/nonzero model update.
- NPU `npu:0`: all eight values completed one `AcousticFWI` iteration with finite loss, finite/nonzero `vp` gradient, and finite/nonzero model update.
- `SmoothL1` and `StudentT` need a larger smoke default learning rate (`1e12`) because the tiny synthetic setup produces gradients small enough that `lr=1e8` is rounded away at the `float32` model scale. This is a smoke-test scaling choice, not a backend limitation.

Representative commands:

```bash
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit NIM
```

## Recommended Immediate Next Step

Use `misfit_backend_smoke.py` for tensor-level gates and `acoustic_mini_inversion_smoke.py --misfit ...` for one-step FWI gates before changing low-risk misfit behavior.

Do not modify `SoftDTW`, `WDGC`, or heavy Wasserstein behavior until the portable group remains stable under both smoke layers.
