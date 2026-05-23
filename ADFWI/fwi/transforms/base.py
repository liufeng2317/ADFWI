"""Composable waveform data transforms for ADFWI FWI workflows."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

Context = Optional[Dict[str, Any]]
TensorPair = Tuple[torch.Tensor, torch.Tensor]


class DataTransform:
    """Base class for transforms that operate on synthetic/observed tensors."""

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        raise NotImplementedError


class DataTransformPipeline(DataTransform):
    """Apply a sequence of data transforms to a synthetic/observed tensor pair."""

    def __init__(self, transforms: Optional[Iterable[DataTransform]] = None) -> None:
        self.transforms: List[DataTransform] = list(transforms or [])

    def append(self, transform: DataTransform) -> None:
        self.transforms.append(transform)

    def __len__(self) -> int:
        return len(self.transforms)

    def __call__(self, synthetic: torch.Tensor, observed: torch.Tensor, context: Context = None) -> TensorPair:
        if synthetic.shape != observed.shape:
            raise ValueError(f"synthetic and observed must have the same shape, got {synthetic.shape} and {observed.shape}")

        for transform in self.transforms:
            synthetic, observed = transform(synthetic, observed, context=context)
        return synthetic, observed
