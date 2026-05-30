# Marmousi2 Acoustic Full-Record Script Validation

This validation case mirrors the original acoustic Marmousi2 example under:

```text
examples/acoustic/01-model-test/01-Marmousi2
```

It is script-only and keeps the full acquisition geometry by default:

- `nx=200`, `nz=88`, `dx=40`, `dz=40`
- `nt=3000`, `dt=0.003`, `f0=5`
- sources at `range(2, nx - 1, 5)`, default `--shots 40`
- receivers at every x grid point, default 200 receivers
- Marmousi2 dataset from `examples/datasets/marmousi2_source`
- initial model smoothing: `gaussian_kernel=6`, `rcv_depth=10`, `mask_extra_depth=2`
- optimizer: Adam, `lr=10`, scheduler `step_size=200`, `gamma=0.75`

Generated artifacts are written under `outputs/` and ignored by git.

## Usage

Dry-run the full 300-iteration workflow:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py all \
  --iterations 300 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --dry-run
```

Run forward modeling first:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py forward \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Run 300 inversion iterations using the forward-generated observed data:

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py inversion10 \
  --iterations 300 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1
```

Use `--output-root <path>` to isolate a run. The inversion stage expects
`waveform/obs_data.npz` to exist under the same output root.
