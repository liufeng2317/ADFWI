# Full-Record Marmousi2 Acoustic Baseline

Date: 2026-05-30

This record freezes the first full-record Marmousi2 acoustic validation result
after the bv1.2 framework cleanup and example synchronization. Use it as the
baseline when comparing future propagator, operator, checkpointing, backend, or
FWI-loop performance changes.

## Baseline Scope

Validation case:

```text
examples/validation/marmousi2_acoustic_full_record/
```

Code anchor:

```text
commit: f996c0d
branch: bv1.2
```

The run validates the main acoustic FWI path:

- `ADFWI.set_backend(...)`
- `AcousticModel`
- `Source`, `Receiver`, `Survey`, `SeismicData`
- `AcousticPropagator`
- `AcousticFWI`
- `Misfit_waveform_L2`
- `GradProcessor`
- model/result/figure saving in the script workflow

## Configuration

```text
device: npu:0
dtype: float32
checkpoint_segments: 1
nx, nz: 200, 88
dx, dz: 40, 40
nt, dt: 3000, 0.003
nabc: 30
f0: 5
shots: 40
receivers: 200
free_surface: true
dataset: examples/datasets/marmousi2_source
initial smoothing: gaussian_kernel=6, rcv_depth=10, mask_extra_depth=2
optimizer: Adam, lr=10
scheduler: StepLR(step_size=200, gamma=0.75)
loss: Misfit_waveform_L2(dt=1)
gradient mute: top 12 grid cells
iterations: 300
```

## Commands

Forward:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Inversion:

```bash
setsid bash -lc "cd /liufeng1afs/project/04_Inversion/ADFWI-github && \
  /liufeng1afs/software/miniconda3/bin/conda run -n adfwi python \
  examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py inversion10 \
  --iterations 300 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  > examples/validation/marmousi2_acoustic_full_record/outputs/full_record/logs/inversion300_20260530_182012.log 2>&1" < /dev/null &
```

## Forward Result

Output:

```text
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/forward_summary.json
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/waveform/obs_data.npz
```

Summary:

```text
status: ok
record.p.shape: [40, 3000, 200]
record.p.dtype: float32
record.p.finite: true
record.p.min: -0.04947379231452942
record.p.max: 0.09417726844549179
record.p.norm: 4.604166507720947
seconds: 7.913247490301728
```

## Inversion Result

Output:

```text
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/inversion300_summary.json
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/iter_loss.npz
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/iter_vp.npz
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/loss.png
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/init_vs_inverted_vp.png
examples/validation/marmousi2_acoustic_full_record/outputs/full_record/inversion/true_vs_inverted_vp.png
```

Summary:

```text
status: ok
iterations: 300
initial_loss: 74756.2578125
final_loss: 4211.63671875
loss_reduction: 94.36617503070657 %
min_loss: 2709.09228515625
min_loss_iteration: 283
last10_loss_mean: 3513.262646484375
vp_update_norm: 43990.0078125
seconds: 10125.929992290214
seconds_per_iteration: 33.75309997430071
```

## Baseline Interpretation

This run is the current acoustic full-record baseline for bv1.2. A future
optimization should compare against it before being treated as an improvement.

For numerical behavior, compare at least:

- forward `record.p` shape, finiteness, norm, min, max;
- inversion `initial_loss`, `final_loss`, `min_loss`, and loss-curve trend;
- `vp_update_norm`;
- generated result files and figure availability.

For performance behavior, compare at least:

- forward wall time;
- inversion total wall time;
- average seconds per iteration;
- backend/device/dtype;
- `checkpoint_segments`.

This baseline does not prove full coverage of elastic, DIP, every misfit
function, or every public example. It is specifically the bv1.2 acoustic
Marmousi2 full-record FWI anchor.

