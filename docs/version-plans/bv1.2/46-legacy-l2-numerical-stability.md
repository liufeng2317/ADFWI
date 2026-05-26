# Legacy L2 Numerical Stability

## Problem

`Misfit_waveform_L2` in `ADFWI/fwi/misfit/L2.py` is the historical waveform
L2-norm misfit. Its current objective is:

```text
loss = sum(sqrt(sum((obs - syn)^2 * dt, axis=1)))
```

This is a true norm form, not the squared least-squares waveform objective that
is commonly used as the stable baseline in FWI:

```text
loss = 0.5 * sum((obs - syn)^2 * dt)
```

The norm form has a singular derivative when the residual energy is exactly
zero. In PyTorch autograd, an exactly matched trace or a muted/windowed region
can therefore produce a finite forward loss but a NaN backward gradient.

## Minimal Reproduction

The issue is backend independent. It can be reproduced on CPU with a tiny tensor
where observed and synthetic waveforms are identical:

```bash
conda run -n adfwi python -c "import torch; from ADFWI.fwi.misfit import Misfit_waveform_L2; obs=torch.ones(1,4,1); syn=torch.ones(1,4,1, requires_grad=True); loss=Misfit_waveform_L2(dt=1.0).forward(obs,syn); loss.backward(); print(float(loss)); print(syn.grad); print(torch.isfinite(syn.grad).all().item())"
```

Observed behavior:

```text
loss = 0.0
grad = nan
finite = False
```

## Why This Matters for FWI

The zero-residual case is not artificial in inversion workflows:

- a trace can become exactly matched in a reduced synthetic test;
- a mute, mask, or time window can leave a local zero-residual segment;
- small reduced examples can produce exact zeros after source/receiver
  selection;
- mixed CPU/NPU execution may expose exact-zero cases more often because of
  dtype and kernel differences.

For FWI, a perfectly matched residual should contribute zero gradient, not NaN.
Once NaN enters the model gradient, it can contaminate the optimizer step and
invalidate the inversion.

## Current Decision

Do not silently change `Misfit_waveform_L2` in place yet. Many historical
examples and result comparisons may depend on the old objective scale.

For bv1.2, treat `Misfit_waveform_L2` as a legacy-compatible norm objective:

- backend-portable for ordinary nonzero residual tensors;
- unsafe at exactly zero residual energy;
- not recommended as the default for new CPU/NPU FWI smoke tests.

The reduced Marmousi2 inversion smoke now uses package-level
`Misfit_waveform_SquaredL2(reduction="mean")` through the existing
`safe-squared-l2` command-line option, and keeps `--misfit legacy-l2` only for
diagnostics.

## Next Optimization Design

### 1. Add a Stable Squared-L2 Misfit

Add a new explicit class, for example:

```text
Misfit_waveform_SquaredL2
```

Recommended formula:

```text
loss = 0.5 * sum((obs - syn)^2 * dt)
```

or, if the existing framework expects loss magnitudes normalized by tensor size:

```text
loss = mean((obs - syn)^2) * dt
```

The first formula is closer to classical FWI least-squares notation; the second
is more scale-stable for examples. The implementation should make the reduction
choice explicit through a constructor argument rather than hiding it.

### 2. Keep Legacy L2 Available

Keep `Misfit_waveform_L2` import-compatible. Add a warning in its docstring and
support matrix describing the zero-residual gradient singularity.

Avoid automatic aliasing from `L2` to the new squared version until users have a
clear migration note.

### 3. Add Numerical Tests

Add focused tests for:

- exact match: forward loss finite and gradient finite for the new squared L2;
- nonzero residual: CPU and NPU forward/backward finite;
- zero residual: legacy L2 reproduces the NaN risk, marked as an expected
  diagnostic behavior rather than a passing recommended path;
- shape contract: preserve `[shot, time, receiver]` and component-compatible
  behavior.

### 4. Update Smoke Scripts Gradually

Migrate new smoke tests and script examples to the stable squared L2 default.
Keep a command-line option for legacy L2 diagnostics.

Recommended order:

1. `scripts/examples/marmousi2_acoustic_reduced_inversion.py`
2. `scripts/smoke/acoustic_mini_inversion_smoke.py`
3. `scripts/examples/minimal_acoustic_fwi_backend.py`
4. `scripts/examples/minimal_elastic_fwi_backend.py`

### 5. Validate Numerical Impact

For each migration, compare:

- forward loss finite status;
- model gradient finite status;
- gradient norm and update norm;
- CPU/NPU absolute and relative differences;
- one-step inversion behavior on the tiny synthetic case;
- reduced Marmousi2 inversion behavior.

The acceptance rule should prioritize finite gradients and stable CPU/NPU
agreement. Loss scale differences are acceptable only when documented as a
deliberate objective change.

## Proposed Acceptance Criteria

- Existing legacy imports still work.
- New squared L2 returns zero finite gradient at exact match.
- New squared L2 passes tensor-level CPU/NPU smoke.
- New squared L2 passes acoustic mini inversion smoke on CPU and NPU.
- Reduced Marmousi2 inversion uses the package-level stable L2 rather than a
  local script-only class.
- Documentation clearly distinguishes legacy norm L2 from stable squared L2.


## Implementation Update

A package-level stable squared-L2 misfit was added in bv1.2:

```text
ADFWI.fwi.misfit.Misfit_waveform_SquaredL2
```

The legacy `Misfit_waveform_L2` class remains import-compatible and numerically
unchanged. Its docstring now records the zero-residual square-root singularity.

`Misfit_waveform_SquaredL2` supports two explicit reductions:

- `reduction="sum"`: `0.5 * sum((obs - syn)^2 * dt)` for classical FWI
  least-squares notation;
- `reduction="mean"`: `mean((obs - syn)^2) * dt` for scale-stable smoke and
  reduced example runs.

The reduced Marmousi2 inversion script now uses package-level
`Misfit_waveform_SquaredL2(dt=args.dt_for_loss, reduction="mean")` through the
existing `--misfit safe-squared-l2` option. The local script-only safe loss was
removed.

The tensor-level misfit smoke registry now includes:

```text
SquaredL2, SquaredL2Mean
```

The layered backend smoke suite default `--misfits` value is now:

```text
L2,SquaredL2
```

This keeps legacy coverage while making the stable objective visible in the
standard CPU/NPU check.

## Validation Results

Focused numerical tests:

```bash
conda run -n adfwi python -m unittest tests.test_misfit_squared_l2
```

Result: passed, 5 tests. The tests confirm:

- `Misfit_waveform_SquaredL2` returns zero finite gradient when `obs == syn`;
- `reduction="sum"` matches `0.5 * sum(residual^2 * dt)`;
- `reduction="mean"` matches `mean(residual^2) * dt`;
- legacy `Misfit_waveform_L2` still reproduces the documented NaN-gradient risk
  at exact match.

Tensor-level CPU/NPU smoke:

```bash
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device cpu --misfits SquaredL2,SquaredL2Mean
conda run -n adfwi python scripts/smoke/misfit_backend_smoke.py --device npu:0 --misfits SquaredL2,SquaredL2Mean
```

Results:

| Device | Misfit | Loss | Gradient norm | Status |
| --- | --- | ---: | ---: | --- |
| CPU | `SquaredL2` | `0.0013983859680593014` | `0.0016723552253097296` | OK |
| NPU | `SquaredL2` | `0.0013983859680593014` | `0.0016723553417250514` | OK |
| CPU | `SquaredL2Mean` | `1.4566521713277325e-05` | `1.7420368749299087e-05` | OK |
| NPU | `SquaredL2Mean` | `1.4566521713277325e-05` | `1.7420368749299087e-05` | OK |

Layered backend suite:

```bash
conda run -n adfwi python scripts/smoke/run_backend_smoke_suite.py --suites public,misfit --devices cpu,npu:0
```

Result: passed; `public` and `misfit` suites returned `ok` on CPU and NPU.
The misfit suite covered `L2,SquaredL2` by default.

Acoustic mini inversion with stable squared L2:

```bash
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device cpu --misfit SquaredL2
conda run -n adfwi python scripts/smoke/acoustic_mini_inversion_smoke.py --device npu:0 --misfit SquaredL2
```

Results:

| Device | Loss | `vp_grad_norm` | `vp_update_norm` | Status |
| --- | ---: | ---: | ---: | --- |
| CPU | `7.185453675204873e-14` | `6.891617183472703e-15` | `0.006857701111584902` | OK |
| NPU | `7.18545435283123e-14` | `6.891617606989177e-15` | `0.006857701111584902` | OK |

The mini inversion smoke uses a larger default learning rate for `SquaredL2`
because its gradient scale is much smaller than the historical norm L2 on this
tiny model. This is a smoke-test visibility scale, not a production inversion
recommendation.

Reduced Marmousi2 inversion with package-level squared L2:

```bash
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device npu:0 --shot-count 1 --nt-samples 300
conda run -n adfwi python scripts/examples/marmousi2_acoustic_reduced_inversion.py --device cpu --shot-count 1 --nt-samples 300
```

Results:

| Device | Loss | `vp_grad_norm` | `vp_update_norm` | Status |
| --- | ---: | ---: | ---: | --- |
| NPU | `4.9887585191754624e-06` | `4.905012959926895e-14` | `0.04906421899795532` | OK |
| CPU | `4.9887580644281115e-06` | `4.905012621113716e-14` | `0.04906421899795532` | OK |

The CPU/NPU absolute loss difference is about `4.55e-13`; the gradient norms
match at the expected float32 backend precision level.

## Outcome

The stable squared-L2 objective is now available as a package-level misfit,
covered by focused finite-gradient tests, tensor-level CPU/NPU smoke, acoustic
mini inversion smoke, and reduced Marmousi2 inversion smoke.

This resolves the immediate bv1.2 need for a safe L2-style objective without
changing historical `Misfit_waveform_L2` behavior.
