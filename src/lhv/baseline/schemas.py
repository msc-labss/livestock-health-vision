"""Records at the baseline stage boundary."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..schema import Record, opt, req

__all__ = [
    "BaselineKind",
    "AssessmentState",
    "InjectionShape",
    "BaselineWindow",
    "FeatureBaseline",
    "BaselineSummary",
    "FeatureDeviation",
    "RiskAssessment",
]


class BaselineKind(StrEnum):
    OWN_HISTORY = "own_history"
    HERD = "herd"


class AssessmentState(StrEnum):
    SCORED = "scored"
    INSUFFICIENT_HISTORY = "insufficient_history"


class InjectionShape(StrEnum):
    STEP = "step"
    RAMP = "ramp"
    SPIKE = "spike"
    TRANSIENT = "transient"


@dataclass(frozen=True)
class BaselineWindow(Record):
    """The window a baseline was computed over — enough to recompute it."""

    SCHEMA_NAME = "baseline_window"
    SCHEMA_VERSION = "1"

    kind: BaselineKind = req()
    start: datetime = req()
    end: datetime = req()
    lookback_days: int = req()
    animal_id: str = opt("")
    site_key: str = opt("")
    observation_count: int = opt(0)
    animal_count: int = opt(0)
    excluded_pass_ids: tuple[str, ...] = opt(())

    @property
    def reference(self) -> str:
        subject = self.animal_id or self.site_key
        return (
            f"{self.kind}:{subject}:{self.start.isoformat()}..{self.end.isoformat()}"
            f":lookback={self.lookback_days}d:n={self.observation_count}"
        )


@dataclass(frozen=True)
class FeatureBaseline:
    """Centre and spread of one feature over one window."""

    feature: str
    centre: float
    spread: float
    count: int
    spread_kind: str = "mad"


@dataclass(frozen=True)
class BaselineSummary(Record):
    SCHEMA_NAME = "baseline_summary"
    SCHEMA_VERSION = "1"

    window: BaselineWindow = req()
    features: tuple[FeatureBaseline, ...] = opt(())

    def get(self, feature: str) -> FeatureBaseline | None:
        for baseline in self.features:
            if baseline.feature == feature:
                return baseline
        return None

    @property
    def kind(self) -> BaselineKind:
        return self.window.kind


@dataclass(frozen=True)
class FeatureDeviation:
    """How far one feature sat from each baseline, in robust standard units."""

    feature: str
    value: float
    own_history_z: float | None
    herd_z: float | None
    directional_z: float | None
    contributes: bool


@dataclass(frozen=True)
class RiskAssessment(Record):
    """A deviation-based risk score, or an explicit statement that there is none."""

    SCHEMA_NAME = "risk_assessment"
    SCHEMA_VERSION = "1"

    assessment_id: str = req()
    animal_id: str = req()
    observed_at: datetime = req()
    site_key: str = req()
    day_key: str = req()
    state: AssessmentState = req()
    scale: str = req()
    config_digest: str = req()
    pass_id: str = opt("")
    risk_score: float | None = opt(None)
    uncertainty: float | None = opt(None)
    own_history_deviation: float | None = opt(None)
    herd_relative_deviation: float | None = opt(None)
    own_history_window: BaselineWindow | None = opt(None)
    herd_window: BaselineWindow | None = opt(None)
    deviations: tuple[FeatureDeviation, ...] = opt(())
    observations_available: int = opt(0)
    observations_required: int = opt(0)
    shortfall: int = opt(0)
    derived_from_injected: bool = opt(False)
    stub_derived: bool = opt(True)
    note: str = opt("")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.state is AssessmentState.INSUFFICIENT_HISTORY and self.risk_score is not None:
            raise ValueError(
                f"{self.animal_id}: insufficient history must not carry a numeric risk score"
            )
        if self.state is AssessmentState.SCORED and self.risk_score is None:
            raise ValueError(f"{self.animal_id}: a scored assessment must carry a risk score")

    @property
    def scored(self) -> bool:
        return self.state is AssessmentState.SCORED
