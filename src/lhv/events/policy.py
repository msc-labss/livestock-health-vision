"""The threshold policy.

Thresholds are a named, versioned object rather than a constant in the alerting
code, because P2's whole job is to change them against real prevalence. Every
alert records which policy raised it, so a change of policy is visible in the
event stream instead of being an unexplained shift in alarm burden.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..baseline.schemas import AssessmentState, RiskAssessment
from ..config import ResolvedConfig
from .schemas import EventLevel, PolicyIdentity

__all__ = ["ThresholdPolicy"]


@dataclass(frozen=True)
class ThresholdPolicy:
    name: str
    version: str
    alert_threshold: float
    watch_threshold: float

    @classmethod
    def from_config(cls, config: ResolvedConfig) -> ThresholdPolicy:
        return cls(
            name=config.events.threshold_policy_name,
            version=config.events.threshold_policy_version,
            alert_threshold=config.events.alert_threshold,
            watch_threshold=config.events.watch_threshold,
        )

    @property
    def identity(self) -> PolicyIdentity:
        return PolicyIdentity(name=self.name, version=self.version)

    def level_for(self, assessment: RiskAssessment) -> EventLevel:
        """An assessment without a score cannot cross a threshold."""
        if assessment.state is not AssessmentState.SCORED or assessment.risk_score is None:
            return EventLevel.OBSERVATION
        if assessment.risk_score >= self.alert_threshold:
            return EventLevel.ALERT
        if assessment.risk_score >= self.watch_threshold:
            return EventLevel.WATCH
        return EventLevel.OBSERVATION
