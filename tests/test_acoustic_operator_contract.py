import pytest
import torch

from ADFWI.propagator.acoustic_operator import (
    ACOUSTIC_OPERATOR_STORAGE_MODES,
    AcousticOperatorConfig,
    AcousticOperatorInputs,
    CompiledAcousticOperatorUnavailable,
    acoustic_pressure_operator,
    compiled_acoustic_operator_available,
)


def _config(**kwargs):
    defaults = dict(
        nx=8,
        nz=6,
        dx=10.0,
        dz=10.0,
        nt=12,
        dt=0.001,
        nabc=2,
        free_surface=False,
        storage_mode="checkpoint",
    )
    defaults.update(kwargs)
    return AcousticOperatorConfig(**defaults)


def _inputs(config=None):
    config = _config() if config is None else config
    src_n = 2
    rcv_n = 3
    return AcousticOperatorInputs(
        src_x=torch.tensor([1, 2], dtype=torch.long),
        src_z=torch.tensor([1, 1], dtype=torch.long),
        src_v=torch.zeros((src_n, config.nt), dtype=torch.float32),
        rcv_x=torch.tensor([1, 2, 3], dtype=torch.long),
        rcv_z=torch.tensor([1, 1, 1], dtype=torch.long),
        damp=torch.zeros(
            (config.nz + 2 * config.nabc, config.nx + 2 * config.nabc),
            dtype=torch.float32,
        ),
        vp=torch.full((config.nz, config.nx), 2000.0, dtype=torch.float32),
        rho=torch.full((config.nz, config.nx), 1000.0, dtype=torch.float32),
    )


def test_acoustic_operator_contract_accepts_supported_storage_modes():
    for storage_mode in ACOUSTIC_OPERATOR_STORAGE_MODES:
        config = _config(storage_mode=storage_mode)
        _inputs(config).validate(config)


def test_acoustic_operator_contract_rejects_invalid_storage_mode():
    config = _config(storage_mode="cpu")

    with pytest.raises(ValueError, match="storage_mode"):
        config.validate()


def test_acoustic_operator_contract_rejects_non_pressure_mode():
    config = _config(pressure_only=False)

    with pytest.raises(ValueError, match="pressure_only=True"):
        config.validate()


def test_acoustic_operator_contract_validates_wavelet_shape():
    config = _config()
    inputs = _inputs(config)
    bad_inputs = AcousticOperatorInputs(
        src_x=inputs.src_x,
        src_z=inputs.src_z,
        src_v=torch.zeros((inputs.src_x.numel(), config.nt + 1), dtype=torch.float32),
        rcv_x=inputs.rcv_x,
        rcv_z=inputs.rcv_z,
        damp=inputs.damp,
        vp=inputs.vp,
        rho=inputs.rho,
    )

    with pytest.raises(ValueError, match="src_v"):
        bad_inputs.validate(config)


def test_acoustic_operator_contract_validates_index_dtype():
    config = _config()
    inputs = _inputs(config)
    bad_inputs = AcousticOperatorInputs(
        src_x=inputs.src_x.to(torch.float32),
        src_z=inputs.src_z,
        src_v=inputs.src_v,
        rcv_x=inputs.rcv_x,
        rcv_z=inputs.rcv_z,
        damp=inputs.damp,
        vp=inputs.vp,
        rho=inputs.rho,
    )

    with pytest.raises(TypeError, match="src_x"):
        bad_inputs.validate(config)


def test_acoustic_pressure_operator_has_explicit_unavailable_boundary():
    config = _config()
    inputs = _inputs(config)

    assert compiled_acoustic_operator_available() is False
    with pytest.raises(CompiledAcousticOperatorUnavailable):
        acoustic_pressure_operator(config, inputs)
