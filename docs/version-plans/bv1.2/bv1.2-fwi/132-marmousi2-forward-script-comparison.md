# 132 - Marmousi2 Forward Script Comparison

## Optimization Path

Validate that the separated forward script produces the same observed data as
the manually verified forward notebook.

## Change

- Ran `scripts/forward_modeling.py forward` into an isolated comparison output
  directory.
- Fixed the forward script summary path so it summarizes the saved numpy data
  after `SeismicData.record_data()` mutates waveform tensors into arrays.

## Comparison

Reference:

- `examples/validation/marmousi2_acoustic_reduced/outputs/minimal_notebook/waveform/obs_data.npz`

Script output:

- `examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare/waveform/obs_data.npz`

Result:

- Metadata matched exactly:
  - `src_num = 3`;
  - `rcv_num = 200`;
  - `nt = 3000`;
  - `dt = 0.003`;
  - source/receiver locations and types matched exactly.
- Data arrays matched exactly:
  - `p`: relative difference `0.0`, max absolute difference `0.0`;
  - `u`: relative difference `0.0`, max absolute difference `0.0`;
  - `w`: relative difference `0.0`, max absolute difference `0.0`;
  - `forward_wavefield_p/u/w`: relative difference `0.0`, max absolute difference `0.0`.
- Script forward runtime on NPU for the 3-shot case was about `8.04 s`.

## Scientific Contract

- No FWI core code changed.
- Forward numerical output from the script is bitwise identical to the notebook
  reference output for this validation case.

## Validation

Completed validation:

- `conda run -n adfwi python examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py forward --output-root examples/validation/marmousi2_acoustic_reduced/outputs/script_forward_compare --device npu:0 --shots 3 --checkpoint-segments 1`: passed.
- Script-vs-notebook `obs_data.npz` comparison: passed with zero difference.
- `conda run -n adfwi python -m py_compile examples/validation/marmousi2_acoustic_reduced/scripts/forward_modeling.py`: passed.

## Next Direction

Run the separated inversion script against the same forward-generated observed
data and compare loss/update summaries with the manually verified notebook
result.
