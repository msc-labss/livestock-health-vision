"""Own-history and herd baselines, and the risk score computed against them.

Two baselines rather than one, because an animal that looks unusual against its
own past is a different fact from an animal that looks unusual against the herd
standing next to it. A feed change, a wet yard or a heat event moves everybody;
reporting only own-history deviation would call that a herd of lame animals.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import timedelta

from ..config import ResolvedConfig
from ..profiles import SpeciesProfile
from .schemas import (
    AssessmentState,
    BaselineKind,
    BaselineSummary,
    BaselineWindow,
    FeatureBaseline,
    FeatureDeviation,
    RiskAssessment,
)
from .store import Observation, TimeSeriesStore

__all__ = ["BaselineEngine", "robust_z"]

# Scales the median absolute deviation so that, for normally distributed data,
# it estimates the standard deviation.
_MAD_TO_SIGMA = 1.4826


def _spread(values: Sequence[float]) -> tuple[float, str]:
    """A robust spread, falling back to the standard deviation when MAD is zero."""
    if len(values) < 2:
        return 0.0, "undefined"
    centre = statistics.median(values)
    mad = statistics.median([abs(v - centre) for v in values]) * _MAD_TO_SIGMA
    if mad > 0:
        return mad, "mad"
    deviation = statistics.pstdev(values)
    return deviation, "stdev"


def robust_z(value: float, baseline: FeatureBaseline) -> float | None:
    if baseline.spread <= 0:
        return None
    return (value - baseline.centre) / baseline.spread


class BaselineEngine:
    """Builds baselines from the stored series and scores an observation against them."""

    def __init__(
        self, store: TimeSeriesStore, profile: SpeciesProfile, config: ResolvedConfig
    ) -> None:
        self.store = store
        self.profile = profile
        self.config = config

    # -- baselines ----------------------------------------------------------

    def own_history(self, observation: Observation) -> BaselineSummary:
        """The animal's own prior observations, excluding the one being scored.

        Strictly prior: an observation must not contribute to the baseline it is
        measured against, or every observation looks normal by construction.
        """
        end = observation.observed_at
        start = end - timedelta(days=self.config.baseline.lookback_days)
        prior = self.store.read(
            animal_id=observation.animal_id,
            start=start,
            end=end,
            exclude_pass_ids=[observation.pass_id],
        )
        prior = [o for o in prior if o.pass_id != observation.pass_id]
        window = BaselineWindow(
            kind=BaselineKind.OWN_HISTORY,
            start=start,
            end=end,
            lookback_days=self.config.baseline.lookback_days,
            animal_id=observation.animal_id,
            site_key=observation.site_key,
            observation_count=len(prior),
            animal_count=1,
            excluded_pass_ids=(observation.pass_id,),
        )
        return BaselineSummary(window=window, features=self._summarise(prior))

    def herd(self, observation: Observation) -> BaselineSummary:
        """The contemporaneous herd at the same site over the declared window."""
        end = observation.observed_at + timedelta(seconds=1)
        start = observation.observed_at - timedelta(days=self.config.baseline.herd_window_days)
        contemporaries = self.store.read(
            site_key=observation.site_key,
            start=start,
            end=end,
            exclude_pass_ids=[observation.pass_id],
        )
        window = BaselineWindow(
            kind=BaselineKind.HERD,
            start=start,
            end=end,
            lookback_days=self.config.baseline.herd_window_days,
            site_key=observation.site_key,
            observation_count=len(contemporaries),
            animal_count=len({o.animal_id for o in contemporaries}),
            excluded_pass_ids=(observation.pass_id,),
        )
        return BaselineSummary(window=window, features=self._summarise(contemporaries))

    def _summarise(self, observations: list[Observation]) -> tuple[FeatureBaseline, ...]:
        summaries = []
        for name in self.profile.feature_set.names:
            values = [o.features[name] for o in observations if o.features.get(name) is not None]
            if not values:
                continue
            spread, kind = _spread(values)
            summaries.append(
                FeatureBaseline(
                    feature=name,
                    centre=statistics.median(values),
                    spread=spread,
                    count=len(values),
                    spread_kind=kind,
                )
            )
        return tuple(summaries)

    # -- scoring ------------------------------------------------------------

    def assess(self, observation: Observation) -> RiskAssessment:
        own = self.own_history(observation)
        herd = self.herd(observation)
        return self.score(observation, own, herd)

    def score(
        self,
        observation: Observation,
        own: BaselineSummary,
        herd: BaselineSummary,
    ) -> RiskAssessment:
        minimum = self.config.baseline.min_observations
        available = own.window.observation_count
        base = dict(
            assessment_id=f"{observation.pass_id}:risk",
            animal_id=observation.animal_id,
            observed_at=observation.observed_at,
            site_key=observation.site_key,
            day_key=observation.day_key,
            scale=self.config.baseline.risk_scale,
            config_digest=self.config.digest,
            pass_id=observation.pass_id,
            own_history_window=own.window,
            herd_window=herd.window,
            observations_available=available,
            observations_required=minimum,
            derived_from_injected=observation.injected,
        )

        if available < minimum:
            # An animal without history is not a low-risk animal. Saying so
            # numerically would be a fabrication dressed as reassurance.
            return RiskAssessment(
                state=AssessmentState.INSUFFICIENT_HISTORY,
                shortfall=minimum - available,
                note=(
                    f"{available} prior observation(s) within "
                    f"{self.config.baseline.lookback_days} days; {minimum} required"
                ),
                **base,
            )

        deviations = self._deviations(observation, own, herd)
        contributing = [d.directional_z for d in deviations if d.contributes]
        if not contributing:
            return RiskAssessment(
                state=AssessmentState.INSUFFICIENT_HISTORY,
                shortfall=0,
                note="no feature had a usable baseline spread",
                deviations=deviations,
                **base,
            )

        risk = sum(contributing) / len(contributing)
        # Standard error of the mean deviation: the spread of the evidence
        # divided by how much of it there is.
        uncertainty = (
            statistics.pstdev(contributing) / (len(contributing) ** 0.5)
            if len(contributing) > 1
            else float("inf")
        )
        own_deviation = _mean_or_none([d.own_history_z for d in deviations])
        herd_deviation = _mean_or_none([d.herd_z for d in deviations])

        return RiskAssessment(
            state=AssessmentState.SCORED,
            risk_score=risk,
            uncertainty=uncertainty,
            own_history_deviation=own_deviation,
            herd_relative_deviation=herd_deviation,
            deviations=deviations,
            **base,
        )

    def _deviations(
        self, observation: Observation, own: BaselineSummary, herd: BaselineSummary
    ) -> tuple[FeatureDeviation, ...]:
        results = []
        for definition in self.profile.feature_set.features:
            value = observation.features.get(definition.name)
            if value is None:
                continue
            own_baseline = own.get(definition.name)
            herd_baseline = herd.get(definition.name)
            own_z = robust_z(value, own_baseline) if own_baseline else None
            herd_z = robust_z(value, herd_baseline) if herd_baseline else None

            directional = None
            if own_z is not None:
                # Orient every feature so that a larger number means worse,
                # whichever direction the feature itself runs.
                directional = own_z if definition.higher_is_worse else -own_z
            results.append(
                FeatureDeviation(
                    feature=definition.name,
                    value=value,
                    own_history_z=own_z,
                    herd_z=herd_z,
                    directional_z=directional,
                    contributes=directional is not None,
                )
            )
        return tuple(results)

    # -- reproduction -------------------------------------------------------

    def recompute(self, assessment: RiskAssessment) -> RiskAssessment:
        """Reproduce a score from the stored series using only its recorded windows.

        This is what makes the score's recorded inputs a claim that can be
        checked rather than a decoration.
        """
        if assessment.own_history_window is None or assessment.herd_window is None:
            raise ValueError(f"{assessment.assessment_id}: no recorded windows to recompute from")

        observation = self._observation_for(assessment)
        own_window = assessment.own_history_window
        herd_window = assessment.herd_window

        prior = self.store.read(
            animal_id=own_window.animal_id,
            start=own_window.start,
            end=own_window.end,
            exclude_pass_ids=own_window.excluded_pass_ids,
        )
        contemporaries = self.store.read(
            site_key=herd_window.site_key,
            start=herd_window.start,
            end=herd_window.end,
            exclude_pass_ids=herd_window.excluded_pass_ids,
        )
        own = BaselineSummary(window=own_window, features=self._summarise(prior))
        herd = BaselineSummary(window=herd_window, features=self._summarise(contemporaries))
        return self.score(observation, own, herd)

    def _observation_for(self, assessment: RiskAssessment) -> Observation:
        matches = self.store.read(
            animal_id=assessment.animal_id,
            start=assessment.observed_at,
            end=assessment.observed_at + timedelta(seconds=1),
        )
        for candidate in matches:
            if candidate.pass_id == assessment.pass_id:
                return candidate
        raise LookupError(
            f"{assessment.assessment_id}: pass {assessment.pass_id!r} is no longer in the store"
        )


def _mean_or_none(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None
