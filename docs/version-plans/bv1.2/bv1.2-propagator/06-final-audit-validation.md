# 06 - Final Audit and Validation

## Goal

Run a final closeout audit for `ADFWI/propagator` after the non-kernel cleanup
rounds. This record confirms that the current propagator layer has clear
ownership boundaries and that the staged Marmousi2 validation still matches the
existing forward reference output.

No propagator code, kernel formula, boundary formula, checkpoint behavior, or
gradient-processing numerical logic is changed in this record.

## Audit Scope

- `ADFWI/propagator/__init__.py`
- `ADFWI/propagator/acoustic_propagator.py`
- `ADFWI/propagator/elastic_propagator.py`
- `ADFWI/propagator/boundary_condition.py`
- `ADFWI/propagator/gradient_process.py`
- `docs/version-plans/bv1.2-propagator/`
- `examples/validation/marmousi2_acoustic_bv12/`

Kernel files were inspected as part of the package map, but remain out of scope
for readability or formula edits:

- `ADFWI/propagator/acoustic_kernels.py`
- `ADFWI/propagator/acoustic_kernels_bs.py`
- `ADFWI/propagator/elastic_kernels.py`

## Findings

- The public package exports remain concise: `AcousticPropagator`,
  `ElasticPropagator`, `GradProcessor`, and `TorchGradProcessor`.
- Acoustic and elastic propagator wrappers are documented as model/survey/backend
  adapters plus kernel dispatchers.
- `boundary_condition.py` is documented as deterministic NumPy boundary profile
  construction, not wave propagation.
- `gradient_process.py` is documented as FWI gradient post-processing. It remains
  exported from `ADFWI.propagator` for compatibility, but it is not part of the
  forward propagator contract.
- No obvious active-code confusion was found that justifies additional
  non-kernel cleanup.

## Validation Commands

```bash
conda run -n adfwi python -m py_compile ADFWI/propagator/*.py
conda run -n adfwi python -m unittest tests/test_boundary_conditions.py tests/test_torch_grad_processor.py tests/test_backend_integration.py tests/test_marmousi2_validation_example.py
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py check --output-root examples/validation/marmousi2_acoustic_bv12/outputs/propagator_final_audit --device npu:0 --shots 3 --checkpoint-segments 1
conda run -n adfwi python examples/validation/marmousi2_acoustic_bv12/scripts/run_validation.py forward --output-root examples/validation/marmousi2_acoustic_bv12/outputs/propagator_final_audit --device npu:0 --shots 3 --checkpoint-segments 1
git diff --check
```

## Validation Result

- `py_compile` passed for all files in `ADFWI/propagator`.
- Propagator-related unit tests passed: 42 tests.
- Marmousi2 validation `check` passed on `npu:0` with `float32`,
  `shots=3`, `receivers=200`, and `nt=3000`.
- Marmousi2 validation `forward` passed on `npu:0`.
- Forward output summary:
  - pressure data shape: `[3, 3000, 200]`;
  - pressure data dtype: `float32`;
  - pressure data finite: `true`;
  - pressure data norm: `1.3137102127075195`;
  - forward runtime: `7.484241869300604` seconds.

## Forward Comparison

Reference:

```text
examples/validation/marmousi2_acoustic_bv12/outputs/script_forward_compare/waveform/obs_data.npz
```

Candidate:

```text
examples/validation/marmousi2_acoustic_bv12/outputs/propagator_final_audit/waveform/obs_data.npz
```

Comparison result:

| Field | Shape | max_abs | l2_rel | finite |
| --- | --- | ---: | ---: | --- |
| `data["p"]` | `[3, 3000, 200]` | `0.0` | `0.0` | yes |
| `data["u"]` | `[3, 3000, 200]` | `0.0` | `0.0` | yes |
| `data["w"]` | `[3, 3000, 200]` | `0.0` | `0.0` | yes |
| `data["forward_wavefield_p"]` | `[88, 200]` | `0.0` | `0.0` | yes |
| `data["forward_wavefield_u"]` | `[88, 200]` | `0.0` | `0.0` | yes |
| `data["forward_wavefield_w"]` | `[88, 200]` | `0.0` | `0.0` | yes |
| source/receiver metadata | unchanged | `0.0` | `0.0` | yes |

The candidate output is bitwise identical to the existing
`script_forward_compare` validation reference for the compared arrays.

## Stop Rule

Treat `ADFWI/propagator` as closed for framework-level cleanup. Future work
should only enter this package with one of:

- a failing forward, backward, or FWI validation case;
- a dedicated kernel numerical validation plan;
- a measured performance target and benchmark;
- a clear public API or backend-device contract defect.

The next optimization direction should stay outside propagator unless one of
these conditions is met.
