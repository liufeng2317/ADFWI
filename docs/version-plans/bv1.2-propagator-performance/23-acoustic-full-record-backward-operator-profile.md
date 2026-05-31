# 23 - Acoustic Full-Record Backward Operator Profile

Date: 2026-05-31

## Purpose

Confirm whether the earlier acoustic backward operator profile still explains
the full-record shape, without repeating the same reduced-case analysis.

Question:

```text
Does full-record acoustic backward remain dominated by slice/copy/zero/allocation
autograd overhead, or does a different operator become dominant?
```

No propagator kernel behavior was changed.

## Harness Change

Updated:

- `scripts/benchmark/acoustic_backward_operator_profile.py`

Change:

- added configurable `--shots`, `--receivers`, `--source-spacing`,
  `--source-depth`, and `--receiver-depth`;
- changed the profiled `shot_index` from fixed `[0]` to
  `np.arange(args.shots)`;
- preserved the original defaults: `shots=1`, `receivers=3`.

This keeps the previous small operator-profile behavior available while making
full-record batch profiling explicit.

## Command

```bash
timeout 1500s conda run -n adfwi python \
  scripts/benchmark/acoustic_backward_operator_profile.py \
  --device npu:0 \
  --dtype float32 \
  --warmup 0 \
  --repeat 1 \
  --nx 200 \
  --nz 88 \
  --nabc 30 \
  --nt 3000 \
  --shots 40 \
  --receivers 200 \
  --source-spacing 5 \
  --source-depth 1 \
  --receiver-depth 1 \
  --checkpoint-segments 1 \
  --topk 25 \
  --output docs/version-plans/bv1.2-propagator-performance/acoustic_full_record_backward_operator_profile_20260531.json
```

Verification:

```bash
conda run -n adfwi python -m py_compile \
  scripts/benchmark/acoustic_backward_operator_profile.py
```

## Case

| Field | Value |
| --- | --- |
| device / dtype | `npu:0`, `float32` |
| model | `nx=200`, `nz=88`, `nabc=30` |
| time samples | `nt=3000` |
| shots / receivers | `40 / 200` |
| output shape | `[40, 3000, 200]` |
| `checkpoint_segments` | `1` |
| `save_forward_wavefield` | `true` |
| loss component | `p` |

Profiler backward wall time was `122.60 s`. This is profiler-instrumented time
and should not be compared with normal FWI iteration wall time. The useful
result is the operator distribution.

## Numerical Sanity

| Metric | Value |
| --- | ---: |
| loss | `6.1377627424974435e-09` |
| pressure finite | `true` |
| `vp.grad` finite | `true` |
| `vp.grad` norm | `3.039276079072617e-12` |

## Top Operators

Sorted by `self_device_time_total_us`.

| Rank | Operator | Count | Self device time |
| --- | --- | ---: | ---: |
| 1 | `aten::slice_backward` | `179941` | `10.444706 s` |
| 2 | `aclnnInplaceCopy` | `218949` | `10.212467 s` |
| 3 | `autograd::engine::evaluate_function: SliceBackward0` | `179941` | `8.636189 s` |
| 4 | `aten::copy_` | `218951` | `8.598612 s` |
| 5 | `aclnnInplaceZero` | `200943` | `8.489193 s` |
| 6 | `empty_tensor` | `315108` | `7.907318 s` |
| 7 | `aten::zero_` | `200943` | `7.743913 s` |
| 8 | `aten::zeros` | `182943` | `7.651679 s` |
| 9 | `SliceBackward0` | `179941` | `4.741699 s` |
| 10 | `aten::slice` | `179940` | `4.439516 s` |

Approximate grouping over the top-25 self device time:

| Group | Time | Share |
| --- | ---: | ---: |
| zero/allocation | `31.792103 s` | `31.13%` |
| slice/view backward | `28.262110 s` | `27.67%` |
| copy/write/index put | `22.637747 s` | `22.17%` |
| simple math | `13.907636 s` | `13.62%` |
| other | `5.531449 s` | `5.42%` |

## Comparison With Earlier Profile

The earlier small operator profile in
`12-acoustic-default-backward-operator-profile.md` reported the same dominant
families:

- `aten::copy_` / `aclnnInplaceCopy`;
- `aten::slice_backward` / `SliceBackward0`;
- `aten::zero_` / `aten::zeros` / `empty_tensor`;
- `CopySlices` and index-put work.

The full-record batch profile confirms that the bottleneck is not a different
physical math operator at larger shape. It remains the PyTorch autograd pattern
created by many sliced timestep updates.

## Decision

Stop repeating backward operator profiling.

The current conclusion is stable across:

- small operator-profile shape;
- reduced FWI iteration profile;
- full-record FWI iteration profile;
- full-record operator-profile shape.

Next work should be a decision/feasibility step, not another profile:

```text
Decide whether to attempt a bounded acoustic timestep state-update rewrite.
If not, close acoustic kernel micro-optimization and move to a higher-impact
route such as custom adjoint/autograd research or memory policy experiments.
```

Any timestep rewrite must compare:

- forward `p/u/w`;
- raw `vp.grad`;
- reduced FWI loss and model update;
- full-record short validation before acceptance.
