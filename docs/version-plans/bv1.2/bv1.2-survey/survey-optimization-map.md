# Survey Optimization Map

This map shows the intended ownership boundaries for `ADFWI/survey`.

## Layer Map

```mermaid
flowchart TD
    User[User / examples / validation case]
    User --> SurveyAPI[ADFWI.survey public API]

    SurveyAPI --> Source[source.py\nSource]
    SurveyAPI --> Receiver[receiver.py\nReceiver]
    SurveyAPI --> Survey[survey.py\nSurvey]
    SurveyAPI --> Data[data.py\nSeismicData]

    Source --> SourceMeta[time sampling\nnt dt f0 t]
    Source --> SourceGeom[source grid locations\nsrc_x src_z]
    Source --> SourceWavelet[wavelet arrays\nmoment tensors]
    Source --> SourceType[source type\ncurrently mt]

    Receiver --> ReceiverMeta[time sampling\nnt dt]
    Receiver --> ReceiverGeom[receiver grid locations\nrcv_x rcv_z]
    Receiver --> ReceiverType[receiver type\npr vx vz]

    Survey --> Acquisition[combined acquisition geometry]
    Survey --> Masks[optional receiver masks]
    Survey --> SurveyPlots[geometry plotting]

    Data --> Metadata[survey snapshot\nsrc/rcv loc type nt dt]
    Data --> Waveforms[recorded waveform dict]
    Data --> Storage[npz save/load]
    Data --> Parse[acoustic/elastic parse]
    Data --> WaveformPlots[waveform plotting]

    Acquisition --> Propagator[ADFWI.propagator\nforward simulation]
    Waveforms --> FWI[ADFWI.fwi\nloss and inversion]
    Storage --> Examples[examples / validation artifacts]
```

## Responsibility Table

| File | Should Own | Should Not Own |
| --- | --- | --- |
| `source.py` | source time axis, source locations, source wavelets, moment tensors, source type metadata | receiver geometry, recorded data, propagator logic |
| `receiver.py` | receiver time axis, receiver locations, receiver type metadata | source wavelets, recorded data, propagator logic |
| `survey.py` | composition of source/receiver geometry, receiver mask metadata, acquisition plotting | waveform arrays, model data, loss/FWI logic |
| `data.py` | recorded waveform dictionaries, survey metadata snapshot, save/load, acoustic/elastic parse helpers, waveform plots | source/receiver mutation, propagator kernels, FWI loss logic |

## Data And Shape Contracts

```mermaid
flowchart LR
    SourceAdd[Source.add_source/add_sources]
    ReceiverAdd[Receiver.add_receiver/add_receivers]
    SourceAdd --> SourceArrays[get_loc/get_wavelet/get_moment_tensor]
    ReceiverAdd --> ReceiverArrays[get_loc/get_type]
    SourceArrays --> PropagatorInput[Propagator inputs]
    ReceiverArrays --> PropagatorInput
    PropagatorInput --> Record[SeismicData.record_data]
    Record --> NPZ[SeismicData.save npz]
    NPZ --> Load[SeismicData.load]
    Load --> Parse[parse_acoustic_data / parse_elastic_data]
    Parse --> FWI[FWI loss]
```

Current commonly observed contracts:

| Object | Quantity | Expected shape |
| --- | --- | --- |
| `Source.get_loc()` normal source | source locations | `(src_num, 2)` |
| `Source.get_wavelet()` normal source | wavelets | `(src_num, nt)` |
| `Source.get_moment_tensor()` normal source | moment tensors | `(src_num, 3, 3)` |
| `Receiver.get_loc()` | receiver locations | `(rcv_num, 2)` |
| acoustic waveform data | `p`, `u`, `w` | `(src_num, nt, rcv_num)` in validation outputs |
| elastic waveform data | `txx`, `tzz`, `txz`, `vx`, `vz` | `(src_num, nt, rcv_num)` expected by parse/plot paths |
| receiver masks | `receiver_masks` | `(src_num, rcv_num)` |

## Risk Map

| Area | Risk | Validation Needed |
| --- | --- | --- |
| module docstrings/imports/error text | low | `py_compile`, focused construction smoke |
| `__repr__` cleanup | low-medium | empty/non-empty source/receiver/survey tests |
| source/receiver shape conversion | medium | focused shape tests and validation `check` |
| `get_type(unique=True)` ordering | medium | compatibility decision and tests |
| `SeismicData.record_data()` conversion | medium-high | exact waveform dict comparison |
| save/load `.npz` format | high | round-trip compatibility with existing files |
| receiver mask behavior | high | mask tests plus backend integration and forward comparison |
| plot orientation | medium | visual/output smoke only; avoid changing during core cleanup |

## Preferred First Optimization

Start by clarifying names, comments, docstrings, imports, and ownership without
changing array values or saved data format. That gives a stable reading path
before touching `SeismicData` or receiver mask behavior.

## Stop Boundary

Survey optimization should stop once source/receiver/survey/data contracts are
clear and covered by focused tests. Do not use this stage to reshape propagator
inputs, change receiver mask semantics, or migrate all examples.
