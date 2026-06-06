import importlib.util
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ADFWI_ROOT = REPO_ROOT / "ADFWI"
PROPAGATOR_ROOT = ADFWI_ROOT / "propagator"

adfwi_pkg = types.ModuleType("ADFWI")
adfwi_pkg.__path__ = [str(ADFWI_ROOT)]
sys.modules.setdefault("ADFWI", adfwi_pkg)
propagator_pkg = types.ModuleType("ADFWI.propagator")
propagator_pkg.__path__ = [str(PROPAGATOR_ROOT)]
sys.modules.setdefault("ADFWI.propagator", propagator_pkg)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


acoustic_kernels = _load_module(
    "ADFWI.propagator.acoustic_kernels",
    PROPAGATOR_ROOT / "acoustic_kernels.py",
)
acoustic_operator = _load_module(
    "ADFWI.propagator.acoustic_operator",
    PROPAGATOR_ROOT / "acoustic_operator.py",
)

pad_torchSingle = acoustic_kernels.pad_torchSingle
step_forward_pressure_only = acoustic_kernels.step_forward_pressure_only
AcousticPressureSegmentConfig = acoustic_operator.AcousticPressureSegmentConfig
AcousticPressureSegmentInputs = acoustic_operator.AcousticPressureSegmentInputs
CompiledAcousticOperatorUnavailable = acoustic_operator.CompiledAcousticOperatorUnavailable
acoustic_pressure_segment = acoustic_operator.acoustic_pressure_segment


def _make_case(nt=8, shots=2, nx=12, nz=10, nabc=3):
    dtype = torch.float32
    device = torch.device("cpu")
    nx_pml = nx + 2 * nabc
    nz_pml = nz + 2 * nabc
    vp = torch.full((nz, nx), 2200.0, dtype=dtype, device=device)
    rho = torch.full((nz, nx), 2000.0, dtype=dtype, device=device)
    damp = torch.zeros((nz_pml, nx_pml), dtype=dtype, device=device)
    src_x = torch.tensor([nabc + 3, nabc + 7], dtype=torch.long, device=device)
    src_z = torch.full((shots,), nabc + 2, dtype=torch.long, device=device)
    src_index = torch.arange(shots, dtype=torch.long, device=device)
    rcv_x = torch.tensor([nabc + 2, nabc + 5, nabc + 8], dtype=torch.long, device=device)
    rcv_z = torch.full((3,), nabc + 2, dtype=torch.long, device=device)
    src_v = torch.zeros((shots, nt), dtype=dtype, device=device)
    src_v[:, 1] = 1e-6
    p = torch.zeros((shots, nz_pml, nx_pml), dtype=dtype, device=device)
    u = torch.zeros((shots, nz_pml, nx_pml - 1), dtype=dtype, device=device)
    w = torch.zeros((shots, nz_pml - 1, nx_pml), dtype=dtype, device=device)
    c = pad_torchSingle(vp, nabc, nz, nx, shots, device=device)
    den = pad_torchSingle(rho, nabc, nz, nx, shots, device=device)
    return {
        "config": AcousticPressureSegmentConfig(nx=nx, nz=nz, dx=10.0, dz=10.0, dt=0.001, nabc=nabc, free_surface=False),
        "src_x": src_x,
        "src_z": src_z,
        "src_index": src_index,
        "src_v": src_v,
        "rcv_x": rcv_x,
        "rcv_z": rcv_z,
        "kappa1": damp * 0.001,
        "alpha1": den * c.pow(2) * 0.001,
        "kappa2": damp[:, 1:] * 0.001,
        "alpha2": 1.0 / den[:, 1:] * 0.001,
        "kappa3": damp[1:, :] * 0.001,
        "p": p,
        "u": u,
        "w": w,
    }


def _inputs(case, src_v, p=None, u=None, w=None):
    return AcousticPressureSegmentInputs(
        src_x=case["src_x"],
        src_z=case["src_z"],
        src_index=case["src_index"],
        src_v=src_v,
        rcv_x=case["rcv_x"],
        rcv_z=case["rcv_z"],
        kappa1=case["kappa1"],
        alpha1=case["alpha1"],
        kappa2=case["kappa2"],
        alpha2=case["alpha2"],
        kappa3=case["kappa3"],
        p=case["p"] if p is None else p,
        u=case["u"] if u is None else u,
        w=case["w"] if w is None else w,
    )


def test_acoustic_pressure_segment_matches_full_pressure_only_run():
    case = _make_case()
    cfg = case["config"]
    full_p, full_u, full_w, full_rcv_p, _ = step_forward_pressure_only(
        cfg.nx,
        cfg.nz,
        cfg.dx,
        cfg.dz,
        cfg.dt,
        cfg.nabc,
        cfg.free_surface,
        case["src_x"],
        case["src_z"],
        int(case["src_x"].numel()),
        case["src_index"],
        case["src_v"],
        case["rcv_x"],
        case["rcv_z"],
        int(case["rcv_x"].numel()),
        case["kappa1"],
        case["alpha1"],
        case["kappa2"],
        case["alpha2"],
        case["kappa3"],
        9.0 / 8.0,
        -1.0 / 24.0,
        case["p"],
        case["u"],
        case["w"],
        save_forward_wavefield=False,
        accumulate_wavefield_in_grad=False,
        device=case["p"].device,
        dtype=case["p"].dtype,
    )

    first = acoustic_pressure_segment(cfg, _inputs(case, case["src_v"][:, :4]))
    second = acoustic_pressure_segment(
        cfg,
        _inputs(case, case["src_v"][:, 4:], p=first.p, u=first.u, w=first.w),
    )
    chunked_rcv_p = torch.cat((first.rcv_p, second.rcv_p), dim=1)

    assert torch.equal(chunked_rcv_p, full_rcv_p)
    assert torch.equal(second.p, full_p)
    assert torch.equal(second.u, full_u)
    assert torch.equal(second.w, full_w)


def test_acoustic_pressure_segment_compiled_backend_is_explicitly_unavailable():
    case = _make_case(nt=4)
    try:
        acoustic_pressure_segment(case["config"], _inputs(case, case["src_v"]), backend="compiled")
    except CompiledAcousticOperatorUnavailable:
        return
    raise AssertionError("compiled segment backend should be explicitly unavailable")


def test_acoustic_pressure_segment_custom_autograd_forward_matches_reference():
    case = _make_case(nt=4)
    cfg = case["config"]
    inputs = _inputs(case, case["src_v"])
    reference = acoustic_pressure_segment(cfg, inputs, backend="torch_reference")
    candidate = acoustic_pressure_segment(cfg, inputs, backend="custom_autograd_forward")

    assert torch.equal(candidate.rcv_p, reference.rcv_p)
    assert torch.equal(candidate.p, reference.p)
    assert torch.equal(candidate.u, reference.u)
    assert torch.equal(candidate.w, reference.w)


def test_acoustic_pressure_segment_custom_autograd_backward_is_explicitly_unavailable():
    case = _make_case(nt=4)
    case["p"] = case["p"].requires_grad_()
    out = acoustic_pressure_segment(
        case["config"],
        _inputs(case, case["src_v"]),
        backend="custom_autograd_forward",
    )
    try:
        out.rcv_p.sum().backward()
    except RuntimeError as exc:
        assert "backward/gradient policy is not implemented" in str(exc)
        return
    raise AssertionError("custom_autograd_forward segment backward should be unavailable")
