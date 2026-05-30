# Final Global Scan And Smoke Run

## Scope

This is the final verification pass for the `bv1.2-example-sync` branch after
example import, path, and wrapper cleanup.

## Static Scan

Tracked example files checked:

```text
tracked Python files: 633
tracked notebooks: 181
```

Final wrapper/path residual scan:

```text
sys.path.append/sys.path.insert: 0
from ADFWI... import *: 0
matplotlib.use("agg"/"Agg"): 0
tracked notebook path residuals: 0
```

Syntax/format checks:

```text
tracked Python AST parse: passed
tracked notebook JSON parse: passed
```

## Smoke Forward

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/forward_modeling.py forward \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 60 --nz 40 --nt 500 --dx 40 --dz 40 --nabc 20 \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_final_smoke
```

Result:

```text
status=ok
backend=npu:0 float32
seconds=1.6068
record.p.shape=[1, 500, 60]
record.p.finite=true
record.p.norm=0.5580375791
obs_data=examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_final_smoke/waveform/obs_data.npz
```

## Smoke Inversion

Command:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/inversion.py \
  --device npu:0 --dtype float32 --shots 1 --checkpoint-segments 1 \
  --nx 60 --nz 40 --nt 500 --dx 40 --dz 40 --nabc 20 \
  --iterations 2 --lr 10 --gaussian-kernel 4 --rcv-depth 8 \
  --mask-extra-depth 2 --grad-mute-top 8 \
  --output-root examples/validation/marmousi2_acoustic_bv12/outputs/example_sync_final_smoke
```

Result:

```text
status=ok
backend=npu:0 float32
iterations=2
seconds=15.9755
initial_loss=359.28680419921875
final_loss=313.3144836425781
loss_history=[359.28680419921875, 313.3144836425781]
vp_update_norm=675.5803833007812
```

## Interpretation

The representative Marmousi2 validation path still works after the example
cleanup:

- package imports resolve without `sys.path` mutation;
- NPU backend setup still works;
- forward modeling writes finite observed data;
- inversion reloads observed data and performs model update;
- short inversion loss decreases over the two smoke iterations.

The run emitted only known local `torch_npu` Ascend toolkit owner warnings.

## Next Direction

The example-sync branch can be considered structurally closed for tracked
examples. Future work should be case-specific:

- run full or medium-size validation for selected release examples;
- decide separately whether ignored backup/cmp/research notebooks should be
  promoted, archived, or left untouched;
- avoid another broad cleanup pass unless a concrete example category is
  selected.
