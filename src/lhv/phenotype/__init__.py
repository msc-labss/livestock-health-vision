"""Stage boundary: tracklets and poses -> lane passes and locomotion features.

Named, versioned features with per-feature quality flags and an explicit
pass-validity decision.
"""

from .features import FeatureExtractor, KeypointTrack
from .passes import segment_passes
from .schemas import (
    FeatureRecord,
    FeatureValue,
    LanePass,
    PassCompleteness,
    QualityFlag,
    ValidityReason,
)

__all__ = [
    "FeatureExtractor",
    "FeatureRecord",
    "FeatureValue",
    "KeypointTrack",
    "LanePass",
    "PassCompleteness",
    "QualityFlag",
    "ValidityReason",
    "segment_passes",
]
