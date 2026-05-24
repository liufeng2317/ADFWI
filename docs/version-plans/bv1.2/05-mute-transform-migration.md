# bv1.2 Mute Transform Migration

## Goal

Move the waveform mute preprocessing used by `AcousticFWI` and `ElasticFWI` into
the shared data transform pipeline, while preserving the exact legacy numerical
behavior required by FWI tests and examples.

This migration covers:

- offset mute based on source/receiver geometry;
- first-arrival late-window mute;
- default pipeline ordering relative to low-pass filtering, data masks, and trace
  normalization.

## Precision Decision

The first implementation intentionally uses legacy-compatible wrappers instead
of replacing mute logic with a new pure torch/NPU-native implementation. The new
classes delegate to the existing utility functions:

- `LegacyOffsetMute` wraps `ADFWI.utils.offset_mute.mute_offset`;
- `LegacyLateWindowMute` wraps `ADFWI.utils.first_arrivel_picking.apply_mute`.

This keeps the migration structural rather than numerical. A pure torch version
can be added later, but it should be introduced as a separate transform and must
pass explicit waveform/loss/gradient comparisons before becoming the default.

## Pipeline Order

The default `AcousticFWI` and `ElasticFWI` preprocessing pipeline is now:

```python
DataTransformPipeline([
    LegacyOffsetMute(required=False),
    LegacyLateWindowMute(required=False),
    LegacyLowPassFilter(required=False),
    DataMask(required=False, apply_to="synthetic"),
    TraceNormalize(),  # only when waveform_normalize=True
])
```

When users provide a custom `data_transform_pipeline`, ADFWI still prepends the
legacy compatibility transforms so existing arguments remain active:

- `waveform_mute_offset`;
- `waveform_mute_late_window`;
- `cutoff_freq`;
- `obs_data.data_masks`.

The custom pipeline is applied after those compatibility transforms. Existing
`waveform_normalize=True` behavior remains as the final compatibility fallback
when a custom pipeline is supplied.

## Context Contract

The FWI classes pass the following context keys into the transform pipeline:

| Key | Used by | Meaning |
| --- | --- | --- |
| `shot_index` | bookkeeping | active shot indices for the current loss call |
| `receiver_mask` | `LegacyOffsetMute`, `DataMask` | active receiver mask for selected shots |
| `src_x` | `LegacyOffsetMute` | selected source x positions |
| `rcv_x` | `LegacyOffsetMute` | full receiver x position list |
| `dx` | `LegacyOffsetMute` | model grid spacing in x direction |
| `offset_mute_threshold` | `LegacyOffsetMute` | existing `waveform_mute_offset` value |
| `late_window` | `LegacyLateWindowMute` | existing `waveform_mute_late_window` value |
| `cutoff_freq` | `LegacyLowPassFilter` | low-pass cutoff frequency |
| `dt` | `LegacyLateWindowMute`, `LegacyLowPassFilter` | propagator time interval |
| `data_mask` | `DataMask` | selected synthetic sample mask |

## Validation

Unit-level checks added in `tests/test_mute_transform_comparison.py`:

- `LegacyLateWindowMute` output equals the old `apply_mute` loop for synthetic
  and observed tensors;
- `LegacyOffsetMute` output equals the old FWI receiver-selection branch followed
  by `mute_offset`;
- both transforms are no-ops when optional settings are absent and
  `required=False`;
- the complete migrated default preprocessing order
  `offset -> late mute -> legacy low-pass -> data mask -> trace normalize`
  is compared against a hand-reconstructed legacy FWI sequence with
  `torch.equal`, confirming pointwise identical synthetic and observed tensors
  for the real FWI `data_masks` dtype path.

Regression tests run on 2026-05-24 in the `adfwi` conda environment:

```bash
conda run -n adfwi python -m unittest \
  tests/test_mute_transform_comparison.py \
  tests/test_lowpass_transform_comparison.py \
  tests/test_data_transforms.py \
  tests/test_backend_integration.py
```

Result: `Ran 35 tests ... OK`. After adding the full-chain legacy equivalence check, the same command now reports `Ran 36 tests ... OK`.

Subset rerun after replacing the receiver-index helper with an equivalent
warning-free implementation:

```bash
conda run -n adfwi python -m unittest \
  tests/test_mute_transform_comparison.py \
  tests/test_backend_integration.py
```

Result: `Ran 16 tests ... OK`.

Smoke tests after migration:

| Command | Loss | Gradient norm | Update norm | Status |
| --- | ---: | ---: | ---: | --- |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2` | `3.8032811744415085e-07` | `1.8180083216634557e-08` | `1.8180636167526245` | OK |
| `scripts/smoke/elastic_mini_inversion_smoke.py --device cpu` | `0.0008452539914287627` | `4.3503572669578716e-05` | `4.350393295288086` | OK |

These values match the prior no-cutoff smoke baseline, confirming the structural
migration did not change default no-mute behavior.

## Smoke Options

The mini inversion smoke scripts now expose mute controls without writing
notebooks, figures, wavefields, or example outputs:

- `--mute-offset <meters>` passes `waveform_mute_offset`;
- `--mute-late-window <seconds>` passes `waveform_mute_late_window`;
- `--mute-late-window` requires `--nt >= 128` because the legacy
  first-arrival mute path uses a fixed 100-sample taper. Shorter mini records can
  fail inside the legacy function before a meaningful FWI comparison is reached.

CPU smoke baselines recorded on 2026-05-24:

| Command | Loss | Gradient norm | Update norm | Status |
| --- | ---: | ---: | ---: | --- |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2 --mute-offset 15` | `1.2393697179646779e-09` | `1.7858665934955553e-11` | `0.0017605230677872896` | OK |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2 --nt 160 --mute-late-window 0.01` | `2.672405798875843e-07` | `6.492609117003667e-09` | `0.6493018269538879` | OK |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit L2 --nt 160 --mute-offset 15 --mute-late-window 0.01` | `1.7874511115678615e-07` | `2.2286099632395917e-09` | `0.22289079427719116` | OK |
| `scripts/smoke/elastic_mini_inversion_smoke.py --device cpu --mute-offset 15` | `2.0964066607120913e-06` | `6.207965697058171e-08` | `0.006181146949529648` | OK |
| `scripts/smoke/elastic_mini_inversion_smoke.py --device cpu --nt 160 --mute-late-window 0.01` | `0.00045096699614077806` | `1.81546183739556e-05` | `1.8153926134109497` | OK |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2 --mute-offset 15` | `1.2393697179646779e-09` | `1.7858665934955553e-11` | `0.0017605230677872896` | OK |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2 --nt 160 --mute-late-window 0.01` | `2.672405798875843e-07` | `6.492609117003667e-09` | `0.649302065372467` | OK |
| `scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit L2 --nt 160 --mute-offset 15 --mute-late-window 0.01` | `1.7874512536764087e-07` | `2.2286099632395917e-09` | `0.22289079427719116` | OK |
| `scripts/smoke/elastic_mini_inversion_smoke.py --device npu:0 --mute-offset 15` | `2.096416892527486e-06` | `6.207973513028264e-08` | `0.006181146949529648` | OK |
| `scripts/smoke/elastic_mini_inversion_smoke.py --device npu:0 --nt 160 --mute-late-window 0.01` | `0.0004509665013756603` | `1.8154616554966196e-05` | `1.8153928518295288` | OK |

The acoustic CPU/NPU values are identical for offset-only and late-window-only
smokes. The acoustic combined mute and elastic smokes show only float32-scale
last-digit drift in loss, gradient norm, or update norm. This is acceptable for
the current legacy-compatible migration, and the pointwise CPU legacy-equivalence
test remains the primary guard for structural correctness.

## Automated Drift Runner

`ASDFWI` now includes `scripts/smoke/compare_backend_smoke.py` for automated
CPU/NPU smoke drift checks. The runner executes the selected mini inversion
smoke once per device, parses JSON outputs, and compares:

- `loss`;
- `vp_grad_norm`;
- `vp_update_norm`.

A metric passes when either `abs_diff <= --abs-tol` or `rel_diff <= --rel-tol`.
The default tolerances are `--abs-tol 1e-10` and `--rel-tol 1e-5`.

Validated runner commands on 2026-05-24:

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py \
  --problem acoustic --case mute-combined --devices cpu,npu:0
```

Result: `status: ok`. Largest recorded drift was acoustic combined-mute loss
`abs_diff=1.4210854715202004e-14`, `rel_diff=7.950345323249115e-08`.

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py \
  --problem elastic --case mute-late --devices cpu,npu:0
```

Result: `status: ok`. Largest recorded relative drift was elastic late-window
loss `rel_diff=1.0971204589216158e-06`; update norm drift was
`abs_diff=2.384185791015625e-07`, `rel_diff=1.3133167229411938e-07`.

The runner also supports matrix mode with comma-separated `--problems` and
`--cases`:

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py \
  --problems acoustic,elastic --cases baseline,mute-offset --devices cpu,npu:0
```

Result: `status: ok`. This matrix ran 8 child smoke tests. The largest relative
drift was elastic offset loss `rel_diff=4.880620563312549e-06`, still below the
default `1e-5` relative tolerance. Acoustic baseline and acoustic offset both
passed; acoustic offset matched exactly for all three metrics.

Legacy low-pass drift was also validated through matrix mode:

```bash
conda run -n adfwi python scripts/smoke/compare_backend_smoke.py \
  --problems acoustic,elastic --cases legacy-lowpass --devices cpu,npu:0
```

Result: `status: ok`. Acoustic legacy-lowpass max relative drift was
`1.9492708119015068e-07` on `vp_grad_norm`; elastic legacy-lowpass max relative
drift was `4.079844737428672e-07` on `loss`.

## Remaining Work

- Expand automated drift runner coverage to full all-case nightly-style runs.
- Evaluate whether a pure torch/NPU-native mute implementation can match the
  legacy outputs and gradients closely enough for FWI usage.
- Keep trace-missing receiver selection separate from receiver masking until the
  dimension-changing path is fully documented and tested.
