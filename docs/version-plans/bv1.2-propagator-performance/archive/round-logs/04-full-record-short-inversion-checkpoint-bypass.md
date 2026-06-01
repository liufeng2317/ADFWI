# Full-Record Short Inversion After Checkpoint Bypass

Date: 2026-05-30

## Boundary

```text
Goal:
  Verify the acoustic checkpoint_segments=1 bypass in a real full-record FWI
  iteration workflow.

Scope:
  Test and documentation only. No new code changes.

Validation:
  Run 10 full-record Marmousi2 acoustic inversion iterations using the
  checkpoint-bypass branch and compare timing/loss trend against the bv1.2
  baseline record.

Stop:
  Stop after short full-record inversion timing is recorded. Do not start the
  next kernel optimization in this round.
```

## Preconditions

Observed data was generated in the isolated output directory:

```text
examples/validation/marmousi2_acoustic_full_record/outputs/checkpoint_bypass_full_forward/
```

Forward summary:

```text
record.p.shape: [40, 3000, 200]
record.p.dtype: float32
record.p.finite: true
record.p.min: -0.04947379231452942
record.p.max: 0.09417726844549179
record.p.norm: 4.604166507720947
forward seconds: 7.839245941489935
```

The forward numerical summary matches the bv1.2 baseline.

## Command

```bash
conda run -n adfwi python examples/validation/marmousi2_acoustic_full_record/scripts/run_validation.py inversion10 \
  --iterations 10 \
  --device npu:0 \
  --dtype float32 \
  --checkpoint-segments 1 \
  --output-root examples/validation/marmousi2_acoustic_full_record/outputs/checkpoint_bypass_full_forward
```

## Result

```text
status: ok
shots: 40
iterations: 10
checkpoint_segments: 1
initial_loss: 74756.2578125
final_loss: 57679.109375
vp_update_norm: 8813.259765625
seconds: 258.78757610730827
seconds_per_iteration: 25.878757610730826
```

Loss history:

```text
[74756.2578125,
 71691.453125,
 68891.140625,
 66495.21875,
 64582.828125,
 62988.71875,
 61532.8984375,
 60171.9453125,
 58879.83984375,
 57679.109375]
```

The loss decreases monotonically in this 10-iteration full-record test.

## Baseline Comparison

The bv1.2 full 300-iteration baseline is:

```text
seconds_per_iteration: 33.75309997430071
initial_loss: 74756.2578125
final_loss: 4211.63671875
```

The new 10-iteration checkpoint-bypass test is:

```text
seconds_per_iteration: 25.878757610730826
initial_loss: 74756.2578125
final_loss after 10 iterations: 57679.109375
```

Timing interpretation:

- the full-record short inversion is about `23.3%` lower in seconds per
  iteration than the previous 300-iteration average;
- this is not a perfect apples-to-apples comparison because the baseline is a
  300-iteration run and this test is 10 iterations;
- the result is still consistent with the checkpoint-overhead benchmark: the
  real FWI iteration path benefits from bypassing checkpoint when
  `checkpoint_segments == 1`.

Numerical interpretation:

- initial loss is identical to the bv1.2 baseline;
- loss trend is stable and decreasing;
- no NaN/Inf or workflow break occurred.

## Next Direction

The next bounded test should be one of:

1. run a longer full-record inversion, for example 50 or 100 iterations, only if
   we need a stronger performance baseline before the next optimization;
2. move to the next acoustic hot-path candidate, such as avoiding repeated
   timestep-invariant index creation, with the same parity and validation
   matrix.

