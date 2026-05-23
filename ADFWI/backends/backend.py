"""Backend and device dispatch helpers for ADFWI.

This module centralizes CPU/CUDA/NPU device selection for the core ADFWI
physical modeling workflow. It intentionally keeps optional accelerator imports
inside this module so the rest of the package can depend on a small stable API.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union

import torch

DeviceLike = Optional[Union[str, "torch.device"]]
DTypeLike = Union[str, torch.dtype]
PreferLike = Union[str, Iterable[str]]


class BackendError(RuntimeError):
    """Base error for backend resolution failures."""


class BackendUnavailableError(BackendError):
    """Raised when a requested backend is not available."""


@dataclass(frozen=True)
class Backend:
    """Resolved ADFWI execution backend.

    Attributes
    ----------
    name:
        Backend family name: ``cpu``, ``cuda``, or ``npu``.
    device:
        Torch-compatible device object.
    index:
        Device index for accelerator devices, otherwise ``None``.
    dtype:
        Default dtype used by ADFWI tensor factories.
    available:
        Whether the requested backend is available.
    fallback:
        Whether CPU fallback was used.
    reason:
        Optional diagnostic message.
    """

    name: str
    device: torch.device
    index: Optional[int]
    dtype: torch.dtype
    available: bool = True
    fallback: bool = False
    reason: Optional[str] = None

    def resolve_device(self, device: DeviceLike = None) -> torch.device:
        """Resolve a device override relative to this backend."""
        if device is None:
            return self.device
        return _canonical_torch_device(device)

    def to_device(self, value: Any, dtype: Optional[torch.dtype] = None) -> Any:
        """Move tensor-like values to this backend device.

        Non-tensor values are converted with ``torch.as_tensor``.
        """
        target_dtype = self.dtype if dtype is None else dtype
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(device=self.device, dtype=target_dtype)
            return value.to(device=self.device)
        return torch.as_tensor(value, dtype=target_dtype, device=self.device)

    def tensor(self, data: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.as_tensor(data, dtype=self.dtype if dtype is None else dtype, device=self.device)

    def zeros(self, shape: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.zeros(shape, dtype=self.dtype if dtype is None else dtype, device=self.device)

    def ones(self, shape: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.ones(shape, dtype=self.dtype if dtype is None else dtype, device=self.device)

    def empty(self, shape: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.empty(shape, dtype=self.dtype if dtype is None else dtype, device=self.device)

    def arange(self, *args: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return torch.arange(*args, dtype=self.dtype if dtype is None else dtype, device=self.device)

    def synchronize(self) -> None:
        """Synchronize accelerator work if the backend supports it."""
        if self.name == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.index)
        elif self.name == "npu":
            npu = _get_torch_npu_api()
            if npu is not None and _npu_is_available(npu):
                npu.synchronize(self.index)

    def memory_allocated(self) -> Optional[int]:
        """Return allocated accelerator memory in bytes when available."""
        if self.name == "cuda" and torch.cuda.is_available():
            return int(torch.cuda.memory_allocated(self.index))
        if self.name == "npu":
            npu = _get_torch_npu_api()
            if npu is not None and hasattr(npu, "memory_allocated"):
                return int(npu.memory_allocated(self.index))
        return None

    def diagnostics(self) -> Dict[str, Any]:
        """Return backend diagnostics useful for logs and benchmarks."""
        return {
            "name": self.name,
            "device": str(self.device),
            "index": self.index,
            "dtype": str(self.dtype).replace("torch.", ""),
            "available": self.available,
            "fallback": self.fallback,
            "reason": self.reason,
            "memory_allocated": self.memory_allocated(),
        }


_DEFAULT_BACKEND: Optional[Backend] = None
_BACKEND_STACK: List[Backend] = []


def configure_backend(
    device: DeviceLike = None,
    *,
    dtype: DTypeLike = torch.float32,
    fallback: bool = False,
    prefer: PreferLike = ("npu", "cpu"),
) -> Backend:
    """Configure the process-wide default ADFWI backend.

    Parameters
    ----------
    device:
        Requested device. ``None`` enables automatic selection.
    dtype:
        Default dtype for backend tensor factories.
    fallback:
        If ``True``, unavailable explicit accelerator requests fall back to CPU.
        Production inversion scripts should normally keep this ``False``.
    prefer:
        Auto-selection priority used when ``device is None``. The default targets the ADFWI CPU+NPU environment.
    """
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = resolve_backend(device, dtype=dtype, fallback=fallback, prefer=prefer)
    return _DEFAULT_BACKEND


def get_backend(
    device: DeviceLike = None,
    *,
    dtype: Optional[DTypeLike] = None,
    fallback: bool = False,
) -> Backend:
    """Return the current backend or resolve an explicit override.

    This function supports the migration pattern used by constructors:

    ``backend = get_backend(device=device, dtype=dtype)``

    If ``device`` is explicitly provided, a backend for that device is returned
    without changing the process-wide default. If ``device is None``, the current
    configured backend is returned, creating the default backend lazily if needed.
    """
    resolved_dtype = _normalize_dtype(dtype) if dtype is not None else None

    if device is not None:
        return resolve_backend(device, dtype=resolved_dtype or _current_dtype(), fallback=fallback)

    if _BACKEND_STACK:
        current = _BACKEND_STACK[-1]
    else:
        global _DEFAULT_BACKEND
        if _DEFAULT_BACKEND is None:
            _DEFAULT_BACKEND = resolve_backend(None, dtype=resolved_dtype or torch.float32)
        current = _DEFAULT_BACKEND

    if resolved_dtype is not None and current.dtype != resolved_dtype:
        return Backend(
            name=current.name,
            device=current.device,
            index=current.index,
            dtype=resolved_dtype,
            available=current.available,
            fallback=current.fallback,
            reason=current.reason,
        )
    return current


@contextmanager
def use_backend(
    device: DeviceLike = None,
    *,
    dtype: Optional[DTypeLike] = None,
    fallback: bool = False,
) -> Iterator[Backend]:
    """Temporarily set the active ADFWI backend inside a context."""
    backend = get_backend(device=device, dtype=dtype, fallback=fallback) if device is not None else get_backend(dtype=dtype)
    _BACKEND_STACK.append(backend)
    try:
        yield backend
    finally:
        _BACKEND_STACK.pop()


def set_backend(
    device: DeviceLike = None,
    *,
    dtype: DTypeLike = torch.float32,
    fallback: bool = False,
    prefer: PreferLike = ("npu", "cpu"),
) -> Backend:
    """User-facing alias for :func:`configure_backend`."""
    return configure_backend(device, dtype=dtype, fallback=fallback, prefer=prefer)


def backend(
    device: DeviceLike = None,
    *,
    dtype: Optional[DTypeLike] = None,
    fallback: bool = False,
) -> Backend:
    """User-facing alias for :func:`get_backend`."""
    return get_backend(device=device, dtype=_normalize_dtype(dtype) if dtype is not None else None, fallback=fallback)


def backend_diagnostics() -> Dict[str, Any]:
    """Return diagnostics for the active ADFWI backend."""
    return get_backend().diagnostics()


def resolve_backend(
    device: DeviceLike = None,
    *,
    dtype: DTypeLike = torch.float32,
    fallback: bool = False,
    prefer: PreferLike = ("npu", "cpu"),
) -> Backend:
    """Resolve a backend without changing global state."""
    resolved_dtype = _normalize_dtype(dtype)
    resolved_prefer = _normalize_prefer(prefer)

    if device is None:
        name, index = _auto_select(resolved_prefer)
        return Backend(name=name, device=torch.device(name if index is None else f"{name}:{index}"), index=index, dtype=resolved_dtype)

    requested = _parse_device(device)
    name, index = requested

    if name == "cpu":
        return Backend(name="cpu", device=torch.device("cpu"), index=None, dtype=resolved_dtype)

    if name == "cuda":
        if torch.cuda.is_available():
            resolved_index = 0 if index is None else index
            _validate_index("cuda", resolved_index, torch.cuda.device_count())
            return Backend("cuda", torch.device(f"cuda:{resolved_index}"), resolved_index, resolved_dtype)
        return _unavailable_or_fallback("cuda", index, resolved_dtype, fallback, "torch.cuda.is_available() is False")

    if name == "npu":
        npu = _get_torch_npu_api()
        if npu is not None and _npu_is_available(npu):
            resolved_index = 0 if index is None else index
            count = _npu_device_count(npu)
            if count is not None:
                _validate_index("npu", resolved_index, count)
            return Backend("npu", torch.device(f"npu:{resolved_index}"), resolved_index, resolved_dtype)
        return _unavailable_or_fallback("npu", index, resolved_dtype, fallback, "NPU runtime is not available")

    raise BackendError(f"Unsupported backend device: {device!r}")


def _normalize_dtype(dtype: DTypeLike) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        value = dtype.strip().lower().replace("torch.", "")
        dtypes = {
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
            "half": torch.float16,
        }
        try:
            return dtypes[value]
        except KeyError as exc:
            raise BackendError(f"Unsupported backend dtype: {dtype!r}") from exc
    raise BackendError(f"Backend dtype must be str or torch.dtype, got {type(dtype)!r}")


def _normalize_prefer(prefer: PreferLike) -> Tuple[str, ...]:
    if isinstance(prefer, str):
        values = tuple(item.strip().lower() for item in prefer.split(",") if item.strip())
    else:
        values = tuple(str(item).strip().lower() for item in prefer if str(item).strip())
    if not values:
        raise BackendError("Backend preference must include at least one device family")
    for name in values:
        if name not in {"cpu", "cuda", "npu"}:
            raise BackendError(f"Unsupported backend preference: {name!r}")
    return values


def _current_dtype() -> torch.dtype:
    if _BACKEND_STACK:
        return _BACKEND_STACK[-1].dtype
    if _DEFAULT_BACKEND is not None:
        return _DEFAULT_BACKEND.dtype
    return torch.float32


def _auto_select(prefer: Tuple[str, ...]) -> Tuple[str, Optional[int]]:
    for name in prefer:
        if name == "cuda" and torch.cuda.is_available():
            return "cuda", 0
        if name == "npu":
            npu = _get_torch_npu_api()
            if npu is not None and _npu_is_available(npu):
                return "npu", 0
        if name == "cpu":
            return "cpu", None
    return "cpu", None


def _canonical_torch_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    name, index = _parse_device(device)
    if name == "cpu":
        return torch.device("cpu")
    return torch.device(f"{name}:{0 if index is None else index}")


def _parse_device(device: Union[str, torch.device]) -> Tuple[str, Optional[int]]:
    if isinstance(device, torch.device):
        name = device.type.lower()
        return name, device.index

    if not isinstance(device, str):
        raise BackendError(f"Device must be None, str, or torch.device, got {type(device)!r}")

    value = device.strip().lower()
    if not value:
        raise BackendError("Device string must not be empty")

    if ":" in value:
        name, index_text = value.split(":", 1)
        try:
            index = int(index_text)
        except ValueError as exc:
            raise BackendError(f"Invalid device index in {device!r}") from exc
    else:
        name, index = value, None

    aliases = {"gpu": "cuda"}
    name = aliases.get(name, name)
    if name not in {"cpu", "cuda", "npu"}:
        raise BackendError(f"Unsupported backend device: {device!r}")
    if name == "cpu" and index not in {None, 0}:
        raise BackendError(f"CPU device does not support nonzero index: {device!r}")
    return name, index


def _unavailable_or_fallback(name: str, index: Optional[int], dtype: torch.dtype, fallback: bool, reason: str) -> Backend:
    if fallback:
        return Backend("cpu", torch.device("cpu"), None, dtype, available=False, fallback=True, reason=f"Requested {name}:{index or 0}; {reason}")
    raise BackendUnavailableError(f"Requested backend {name}:{index or 0} is unavailable: {reason}")


def _validate_index(name: str, index: int, count: int) -> None:
    if index < 0 or index >= count:
        raise BackendUnavailableError(f"Requested {name}:{index}, but only {count} device(s) are available")


def _get_torch_npu_api() -> Any:
    npu = getattr(torch, "npu", None)
    if npu is not None:
        return npu
    try:
        import torch_npu  # type: ignore  # noqa: F401
    except ImportError:
        return None
    return getattr(torch, "npu", None)


def _npu_is_available(npu: Any) -> bool:
    available = getattr(npu, "is_available", None)
    return bool(available()) if callable(available) else False


def _npu_device_count(npu: Any) -> Optional[int]:
    count = getattr(npu, "device_count", None)
    return int(count()) if callable(count) else None
