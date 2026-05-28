"""Public model regularization objectives."""

from .base import Regularization
from .tikhonov_1order import Tikhonov_1order as regularization_Tikhonov_1order
from .tikhonov_2order import Tikhonov_2order as regularization_Tikhonov_2order
from .tv_1order import TV_1order             as regularization_TV_1order
from .tv_2order import TV_2order             as regularization_TV_2order

__all__ = [
    "Regularization",
    "regularization_Tikhonov_1order",
    "regularization_Tikhonov_2order",
    "regularization_TV_1order",
    "regularization_TV_2order",
]
