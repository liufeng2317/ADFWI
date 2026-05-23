__version__ = "0.1.0"
__author__ = "Liu Feng"

from .backends import backend, backend_diagnostics, get_backend, set_backend

__all__ = [
    "__version__",
    "__author__",
    "backend",
    "backend_diagnostics",
    "get_backend",
    "set_backend",
]
