# Marmousi2 Example Sync Smoke Run

## Scope

This run moved from static import cleanup to a representative executable
validation. It used the existing Marmousi2 acoustic validation scripts:

```text
examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py
examples/validation/marmousi2_acoustic_bv12/scripts/inversion.py
```

The run used a reduced grid and reduced iterations to validate the example
sync path quickly. It is not a numerical benchmark.

## Forward Command

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py forward \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 60 --nz 40 --nt 500 --dx 40 --dz 40 --nabc 20 \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_smoke
```

## Forward Result

```text
status=ok
backend=npu:0 float32
seconds=1.6449
record.p.shape=[1, 500, 60]
record.p.finite=true
record.p.norm=0.5580375791
obs_data=examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_smoke/waveform/obs_data.npz
```

## Inversion Command

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/inversion.py \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 60 --nz 40 --nt 500 --dx 40 --dz 40 --nabc 20 \
  --iterations 2 --lr 10 --gaussian-kernel 4 --rcv-depth 8 \
  --mask-extra-depth 2 --grad-mute-top 8 \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_smoke
```

## Inversion Result

```text
status=ok
backend=npu:0 float32
iterations=2
seconds=16.2600
initial_loss=359.28680419921875
final_loss=313.3144836425781
loss_history=[359.28680419921875, 313.3144836425781]
vp_update_norm=675.5803833007812
```

## Interpretation

The smoke run verified the synchronized example path across:

- package imports after import cleanup;
- backend setup on NPU;
- Marmousi2 model loading and resampling;
- forward modeling and observed-data save;
- observed-data reload in inversion;
- FWI construction, loss evaluation, gradient processing, and model update.

Outputs are under `examples/validation/marmousi2_acoustic_bv12/outputs/`,
which is ignored by the validation case `.gitignore`.

## Next Direction

Run one larger but still bounded Marmousi2 check only if needed, for example
`shots=3`, `iterations=10`, and the validation default grid. Otherwise the
example-sync branch is ready to shift from cleanup to targeted case-level
testing.
