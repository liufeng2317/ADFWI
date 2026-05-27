# 68 - Torch Gradient Processor

## Optimization Path

Continue from the benchmark scaffold by reducing one known accelerator bottleneck:
the legacy gradient processor requires parameter gradients and forward
illumination fields to move through CPU NumPy before being written back to the
model parameter. This step adds an explicit torch-native gradient processor
without changing the default legacy GradProcessor behavior.

## Change

- Added TorchGradProcessor as an opt-in subclass of GradProcessor.
- Added torch-native gradient taper, mask, illumination preconditioning,
  smoothing, and normalization helpers.
- Updated FWI runtime gradient dispatch so processors with forward_torch operate
  directly on parameter.grad tensors and preserve propagator dtype/device.
- Kept legacy GradProcessor on the historical NumPy/SciPy path.
- Exported TorchGradProcessor from ADFWI.propagator.

## Scientific Contract

- Existing scripts that instantiate GradProcessor keep the legacy CPU NumPy
  behavior and numerical contract.
- TorchGradProcessor is opt-in; users must request it explicitly.
- Runtime dispatch changes only select the torch-native path when the processor
  exposes forward_torch.
- Because this touches FWI gradient post-processing, numerical precision was
  tested directly against the legacy GradProcessor for representative settings:
  normalization-only and marine mute plus mask plus normalization. Both tests use
  exact receiver/model-shaped tensors and assert allclose at tight tolerances.
- Existing runtime and backend integration tests were run to guard legacy FWI
  construction and gradient-dispatch behavior.

## Validation

Completed validation in the adfwi conda environment:

- conda run -n adfwi python -m unittest tests/test_torch_grad_processor.py: 4 tests passed.
- conda run -n adfwi python -m unittest tests/test_fwi_runtime.py: 24 tests passed.
- conda run -n adfwi python -m unittest tests/test_backend_integration.py: 22 tests passed.
- conda run -n adfwi python -m py_compile ADFWI/propagator/gradient_process.py ADFWI/fwi/runtime/gradient.py ADFWI/propagator/__init__.py: passed.

The new numerical tests compare TorchGradProcessor output against legacy
GradProcessor output with rtol/atol of 1e-6 for float32 paths and 1e-12 for the
float64 runtime-dispatch path.

## Next Optimization Direction

Benchmark TorchGradProcessor inside the acoustic mini-inversion path on CPU and
NPU. If drift remains controlled, add a user-facing script option that switches
minimal acoustic/elastic examples from GradProcessor to TorchGradProcessor for
accelerator-native gradient post-processing.
