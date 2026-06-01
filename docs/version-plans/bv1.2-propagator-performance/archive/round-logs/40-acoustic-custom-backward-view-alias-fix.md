# Acoustic Custom Backward View-Alias Fix

Date: 2026-05-31

## Problem

The observed-upstream debug started to become repetitive because the failing
condition had already been narrowed enough:

```text
outputs/loss match, but custom backward gradients diverge when upstream reaches
the w-update adjoint path.
```

The key code was in
`scripts/benchmark/acoustic_custom_multistep_update_probe.py`, inside
`CustomTimestepUpdateWithFeatures.backward()`.

The buggy pattern was:

```python
gw = grad_w[:, zw, xw]
grad_w[:, zw, xw] = gw * (1.0 - kappa3[zw, xw])
...
grad_alpha2[zw, xw] += torch.sum(-div_w * gw, dim=0)
grad_div_w = -alpha2[zw, xw].unsqueeze(0) * gw
```

`gw` was a view of `grad_w`. After writing back to `grad_w[:, zw, xw]`, the
later uses of `gw` no longer used the original upstream gradient of the
overwritten `w_new` region; they used the already-scaled value.

## Fix

Clone `gw` before writing to `grad_w`:

```python
gw = grad_w[:, zw, xw].clone()
```

This is still benchmark-only custom-gradient prototype code. No production
propagator code was changed.

## Verification

| Test | Before | After |
| --- | ---: | ---: |
| Single-step `u,w` upstream, CPU float64 | `1.475179603723821e-05` | `1.734723475976807e-18` |
| 2-step observed-scale `rcv_p` upstream, CPU float64 | `1.3612220087472204e-05` | `3.552713678800501e-15` |
| 300-step observed-scale `rcv_p` upstream, CPU float64 | `5.5730345067313465` | `1.8189894035458565e-11` |
| 300-step observed-scale `rcv_p` upstream, NPU float32 | `5.109375` | `0.005859375` |
| Full observed-upstream direct replay, NPU float32 | `0.0003633449086919427` | `2.896413207054138e-07` |

All listed tests kept output and loss differences at `0.0`.

The NPU float32 300-step local recurrence still has a small absolute gradient
difference under large upstream scale, but the CPU float64 formula-level tests
now pass to machine precision. The full observed-upstream direct replay raw
`vp.grad` difference dropped by roughly `1254x`.

## Invalid Check

A tiny FWI-style iteration parity run with `nx=64`, `nz=32`, `nt=120`, and
waveform normalization produced non-finite raw gradients in the production
reference itself. That run is not used as an acceptance gate for this fix.

## Decision

The immediate custom backward formula bug was the `gw` view-aliasing issue.
The benchmark-only custom recurrence is substantially closer to production
autograd after the fix, but it is still not production-ready.

## Next Direction

Continue with acoustic-only custom-gradient validation in this order:

1. Run a production-interface direct kernel parity on validation geometry after
   the `gw` fix.
2. If raw `vp.grad` remains within acceptable float32 tolerance, repeat the
   reduced FWI observed-pressure parity with a finite-gradient configuration.
3. Only then consider a production-facing custom backward prototype.
