"""Unified backend and device dispatch interface for ADFWI."""

from .backend import (
    Backend,
    BackendError,
    BackendUnavailableError,
    backend,
    backend_diagnostics,
    configure_backend,
    get_backend,
    resolve_backend,
    set_backend,
    use_backend,
)

__all__ = [
    "Backend",
    "BackendError",
    "BackendUnavailableError",
    "backend",
    "backend_diagnostics",
    "configure_backend",
    "get_backend",
    "resolve_backend",
    "set_backend",
    "use_backend",
]
