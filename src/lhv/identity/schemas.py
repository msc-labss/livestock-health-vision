"""Records at the identity-anchoring stage boundary.

An assignment records how it was reached, not only what it concluded. Method,
confidence, assignment time and the evidence it rests on are all part of the
record, because an identity that cannot be audited is an identity that cannot be
trusted to carry a longitudinal baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..schema import Record, opt, req

__all__ = [
    "AssignmentMethod",
    "UnresolvedReason",
    "AnchorRecord",
    "IdentityAssignment",
    "IdentityConflict",
]


class AssignmentMethod(StrEnum):
    """How an identity was reached. Consumers filter on this."""

    EXTERNAL_ANCHOR = "external_anchor"
    VISUAL_FALLBACK = "visual_fallback"
    UNRESOLVED = "unresolved"


class UnresolvedReason(StrEnum):
    NO_CANDIDATE = "no_candidate"
    AMBIGUOUS_ANCHOR = "ambiguous_anchor"
    BELOW_CONFIDENCE_FLOOR = "below_confidence_floor"
    CONFLICT = "identity_conflict"


@dataclass(frozen=True)
class AnchorRecord(Record):
    """An external identifier observed at a place and time.

    In production this is a parlour, AMS or RFID reading. In P0 the dataset's
    own ground-truth identity is routed through this same record, marked with
    its anchor source, so the control flow being exercised is the production
    one.
    """

    SCHEMA_NAME = "anchor_record"
    SCHEMA_VERSION = "1"

    animal_id: str = req()
    anchor_source: str = req()
    site_key: str = req()
    observed_from: datetime = req()
    observed_to: datetime = req()
    camera_id: str = opt("")
    reader_id: str = opt("")
    confidence: float = opt(1.0)

    def overlaps(self, start: datetime, end: datetime, *, tolerance_seconds: float = 0.0) -> bool:
        from datetime import timedelta

        slack = timedelta(seconds=tolerance_seconds)
        return (self.observed_from - slack) <= end and start <= (self.observed_to + slack)


@dataclass(frozen=True)
class IdentityAssignment(Record):
    """The outcome of resolving one tracklet, resolved or not."""

    SCHEMA_NAME = "identity_assignment"
    SCHEMA_VERSION = "1"

    tracklet_id: str = req()
    method: AssignmentMethod = req()
    assigned_at: datetime = req()
    site_key: str = req()
    day_key: str = req()
    animal_id: str = opt("")
    confidence: float = opt(0.0)
    anchor_source: str = opt("")
    evidence_reference: str = opt("")
    unresolved_reason: UnresolvedReason | None = opt(None)
    candidate_animal_ids: tuple[str, ...] = opt(())

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.resolved and not self.animal_id:
            raise ValueError(
                f"assignment for {self.tracklet_id!r} claims method {self.method} but names no "
                f"animal"
            )
        if not self.resolved and self.animal_id:
            raise ValueError(
                f"assignment for {self.tracklet_id!r} is unresolved but names animal "
                f"{self.animal_id!r}; an unresolved tracklet must not carry a provisional identity"
            )

    @property
    def resolved(self) -> bool:
        return self.method is not AssignmentMethod.UNRESOLVED

    @property
    def enters_time_series(self) -> bool:
        """Only a resolved assignment contributes to a per-animal series."""
        return self.resolved


@dataclass(frozen=True)
class IdentityConflict(Record):
    """Two tracklets overlapping in time resolved to one animal."""

    SCHEMA_NAME = "identity_conflict"
    SCHEMA_VERSION = "1"

    animal_id: str = req()
    tracklet_ids: tuple[str, ...] = req()
    detected_at: datetime = req()
    site_key: str = req()
    day_key: str = req()
    note: str = opt("")
