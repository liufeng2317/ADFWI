"""Default FWI data transform pipeline construction."""

from __future__ import annotations

from typing import Optional

from ADFWI.fwi.transforms import (
    DataMask,
    DataTransformPipeline,
    LegacyLateWindowMute,
    LegacyLowPassFilter,
    LegacyOffsetMute,
    TraceNormalize,
)


def build_fwi_data_transform_pipeline(
    data_transform_pipeline: Optional[DataTransformPipeline],
    waveform_normalize: bool,
) -> tuple[DataTransformPipeline, bool]:
    """Build the legacy-compatible default FWI data transform pipeline.

    AcousticFWI and ElasticFWI share the same pre-loss transform order. When a
    custom pipeline is provided, it is appended after the legacy-compatible
    offset mute, late-window mute, low-pass filter, and synthetic data mask.
    """

    offset_mute = LegacyOffsetMute(required=False)
    late_mute = LegacyLateWindowMute(required=False)
    lowpass = LegacyLowPassFilter(required=False)
    data_mask = DataMask(required=False, apply_to="synthetic")
    transforms = [offset_mute, late_mute, lowpass, data_mask]

    if data_transform_pipeline is not None:
        return DataTransformPipeline(transforms + [data_transform_pipeline]), waveform_normalize

    if waveform_normalize:
        transforms.append(TraceNormalize())
        waveform_normalize = False
    return DataTransformPipeline(transforms), waveform_normalize
