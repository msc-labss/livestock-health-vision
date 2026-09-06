"""Records at the gait-phenotype stage boundary.

A pass is the unit of measurement, and a feature record is what leaves this
stage. Both carry enough for a downstream consumer to know not just the value
but how much the value is worth: a per-feature quality flag, and an explicit
validity decision on the pass as a whole.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..schema import Record, opt, req

__all__ = [
    "PassCompleteness",
    "QualityFlag",
    "ValidityReason",
    "LanePass",
    "FeatureValue",
    "FeatureRecord",
]


class PassCompleteness(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class QualityFlag(StrEnum):
    """How much a feature's value is worth, given the keypoints behind it."""

    GOOD = "good"
    REDUCED = "reduced"
    UNUSABLE = "unusable"
    # Declared by the profile but not computable under the current skeleton or
    # backend. Distinct from UNUSABLE, which describes a pass that went badly:
    # this describes the configuration, is the same for every pass, and so must
    # not count toward the pass validity decision.
    UNAVAILABLE = "unavailable"


class ValidityReason(StrEnum):
    VALID = "valid"
    PARTIAL_PASS = "partial_pass"
    TOO_FEW_FRAMES = "too_few_frames"
    TOO_MANY_REDUCED_FEATURES = "too_many_reduced_features"
    NO_USABLE_FEATURES = "no_usable_features"
    TOO_FEW_USABLE_FEATURES = "too_few_usable_features"
    UNRESOLVED_IDENTITY = "unresolved_identity"


@dataclass(frozen=True)
class LanePass(Record):
    """One traverse of the lane by one tracklet."""

    SCHEMA_NAME = "lane_pass"
    SCHEMA_VERSION = "1"

    pass_id: str = req()
    tracklet_id: str = req()
    source_id: str = req()
    camera_id: str = req()
    site_key: str = req()
    animal_set_key: str = req()
    day_key: str = req()
    first_frame_index: int = req()
    last_frame_index: int = req()
    completeness: PassCompleteness = req()
    entry_frame_index: int | None = opt(None)
    exit_frame_index: int | None = opt(None)
    first_timestamp: datetime | None = opt(None)
    last_timestamp: datetime | None = opt(None)
    frame_count: int = opt(0)
    direction: str = opt("")
    missing_boundary: str = opt("")

    @property
    def timestamp(self) -> datetime | None:
        """The pass's own observation time: where it lands in a time series."""
        return self.first_timestamp


@dataclass(frozen=True)
class FeatureValue:
    """One named feature, its value, its unit and what its quality rests on."""

    name: str
    value: float
    unit: str
    quality: QualityFlag
    limiting_keypoint: str = ""
    coverage: float = 1.0
    note: str = ""


@dataclass(frozen=True)
class FeatureRecord(Record):
    """The locomotion features of one pass, under a named feature-set version."""

    SCHEMA_NAME = "feature_record"
    SCHEMA_VERSION = "1"

    pass_id: str = req()
    feature_set_version: str = req()
    skeleton_id: str = req()
    skeleton_version: str = req()
    site_key: str = req()
    day_key: str = req()
    valid: bool = req()
    validity_reason: ValidityReason = req()
    features: tuple[FeatureValue, ...] = opt(())
    tracklet_id: str = opt("")
    animal_id: str = opt("")
    observed_at: datetime | None = opt(None)
    completeness: PassCompleteness = opt(PassCompleteness.PARTIAL)
    injected: bool = opt(False)

    def value(self, name: str) -> float | None:
        for feature in self.features:
            if feature.name == name:
                return feature.value
        return None

    def quality(self, name: str) -> QualityFlag | None:
        for feature in self.features:
            if feature.name == name:
                return feature.quality
        return None

    def measurements(self) -> tuple[FeatureValue, ...]:
        """The features a downstream consumer may treat as measurements.

        An invalid pass presents none, whatever was computed for audit. Nor does
        an unavailable feature: it carries no value, and letting its placeholder
        reach a consumer would substitute a default for a measurement that was
        never made.
        """
        if not self.valid:
            return ()
        return tuple(
            f
            for f in self.features
            if f.quality not in (QualityFlag.UNUSABLE, QualityFlag.UNAVAILABLE)
        )

    def as_mapping(self) -> dict[str, float]:
        return {f.name: f.value for f in self.measurements()}
