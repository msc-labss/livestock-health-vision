"""The canonical health event — the contract that leaves the system.

Everything a consumer needs travels inside the event: which animal, when, over
what window, what was measured, how confident, which sensors and models
produced it, which site it came from, and which version of this schema it
conforms to. A consumer must never need out-of-band knowledge to read one.

The P0 markers are part of the contract, not a footnote. An event whose risk
assessment came from the stubbed inference path says so on its face, and says
that it is not a clinical finding.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..schema import Record, opt, req

__all__ = [
    "EventLevel",
    "ObservationWindow",
    "PolicyIdentity",
    "EvidenceClip",
    "HealthEvent",
]


class EventLevel(StrEnum):
    OBSERVATION = "observation"
    WATCH = "watch"
    ALERT = "alert"


@dataclass(frozen=True)
class ObservationWindow:
    """The interval the event's measurements were taken over."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("observation window ends before it starts")

    @property
    def seconds(self) -> float:
        return (self.end - self.start).total_seconds()


@dataclass(frozen=True)
class PolicyIdentity:
    """Which threshold policy raised this, and which version of it."""

    name: str
    version: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class EvidenceClip:
    """A retained clip, or an explicit record that retention did not happen."""

    retained: bool
    path: str = ""
    seconds_before: float = 0.0
    seconds_after: float = 0.0
    masking_applied: bool = False
    masking_mode: str = ""
    frame_count: int = 0
    failure_reason: str = ""


@dataclass(frozen=True)
class HealthEvent(Record):
    SCHEMA_NAME = "health_event"
    SCHEMA_VERSION = "1"

    event_id: str = req()
    animal_id: str = req()
    event_timestamp: datetime = req()
    observation_window: ObservationWindow = req()
    phenotype: dict[str, float] = req()
    confidence: float = req()
    contributing_sensors: tuple[str, ...] = req()
    model_identities: dict[str, str] = req()
    site_key: str = req()
    level: EventLevel = req()
    threshold_policy: PolicyIdentity = req()
    config_digest: str = req()

    day_key: str = opt("")
    pass_id: str = opt("")
    risk_score: float | None = opt(None)
    uncertainty: float | None = opt(None)
    own_history_deviation: float | None = opt(None)
    herd_relative_deviation: float | None = opt(None)
    own_history_window: str = opt("")
    herd_window: str = opt("")
    assessment_state: str = opt("")
    evidence_clip: EvidenceClip | None = opt(None)
    derived_from_injected: bool = opt(False)
    stub_derived: bool = opt(True)
    non_clinical: bool = opt(True)
    idempotency_key: str = opt("")
    notes: tuple[str, ...] = opt(())

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.stub_derived and not self.non_clinical:
            raise ValueError(
                f"event {self.event_id}: a stub-derived event may not be presented as a clinical "
                f"finding"
            )
        if not self.idempotency_key:
            object.__setattr__(self, "idempotency_key", self.compute_idempotency_key())

    def compute_idempotency_key(self) -> str:
        """A key that is the same for the same event and different for a different one."""
        material = "|".join(
            [
                self.animal_id,
                self.event_timestamp.isoformat(),
                self.pass_id,
                str(self.threshold_policy),
                self.config_digest,
                str(self.level),
            ]
        )
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]

    @property
    def is_alert(self) -> bool:
        return self.level is EventLevel.ALERT
