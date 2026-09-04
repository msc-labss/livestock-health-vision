"""Assembling a health event from a risk assessment.

The event is where the P0 disclosure lands. Every event this change produces is
marked stub-derived and non-clinical, because the inference behind it was
exercised against injected deviations rather than fitted to clinical outcomes.
A consumer that ignores the marker is making a choice; a consumer that never
saw one was misled.
"""

from __future__ import annotations

from ..baseline.schemas import RiskAssessment
from ..config import ResolvedConfig
from ..phenotype.schemas import FeatureRecord
from .policy import ThresholdPolicy
from .schemas import EvidenceClip, HealthEvent, ObservationWindow

__all__ = ["EventBuilder"]


class EventBuilder:
    def __init__(
        self,
        config: ResolvedConfig,
        policy: ThresholdPolicy,
        *,
        contributing_sensors: tuple[str, ...] = ("rgb-camera",),
        model_identities: dict[str, str] | None = None,
    ) -> None:
        self.config = config
        self.policy = policy
        self.contributing_sensors = contributing_sensors
        self.model_identities = model_identities or {
            role: str(identity) for role, identity in config.models.items()
        }

    def build(
        self,
        assessment: RiskAssessment,
        record: FeatureRecord,
        *,
        window: ObservationWindow,
        evidence_clip: EvidenceClip | None = None,
    ) -> HealthEvent:
        level = self.policy.level_for(assessment)
        notes = [
            "P0: health inference is stubbed. This event is not clinical evidence.",
        ]
        if assessment.derived_from_injected:
            notes.append("Derived from a time series containing a declared synthetic deviation.")
        if not assessment.scored:
            notes.append(f"No risk score: {assessment.note}")

        return HealthEvent(
            event_id=f"{assessment.assessment_id}:event",
            animal_id=assessment.animal_id,
            event_timestamp=assessment.observed_at,
            observation_window=window,
            phenotype=dict(record.as_mapping()),
            confidence=(
                0.0
                if assessment.uncertainty is None or assessment.uncertainty == float("inf")
                else max(0.0, 1.0 - min(assessment.uncertainty, 1.0))
            ),
            contributing_sensors=self.contributing_sensors,
            model_identities=dict(self.model_identities),
            site_key=assessment.site_key,
            level=level,
            threshold_policy=self.policy.identity,
            config_digest=assessment.config_digest,
            day_key=assessment.day_key,
            pass_id=assessment.pass_id,
            risk_score=assessment.risk_score,
            uncertainty=(
                None
                if assessment.uncertainty is None or assessment.uncertainty == float("inf")
                else assessment.uncertainty
            ),
            own_history_deviation=assessment.own_history_deviation,
            herd_relative_deviation=assessment.herd_relative_deviation,
            own_history_window=(
                assessment.own_history_window.reference if assessment.own_history_window else ""
            ),
            herd_window=(assessment.herd_window.reference if assessment.herd_window else ""),
            assessment_state=str(assessment.state),
            evidence_clip=evidence_clip,
            derived_from_injected=assessment.derived_from_injected,
            stub_derived=True,
            non_clinical=True,
            notes=tuple(notes),
        )
