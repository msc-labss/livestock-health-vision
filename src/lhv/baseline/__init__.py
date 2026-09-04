"""Stage boundary: feature records -> per-animal time series and risk scores.

Own-history and herd baselines, an explicit cold-start state, and deviation
scoring with uncertainty.
"""

from .baselines import BaselineEngine, robust_z
from .injection import InjectionSpec, inject, summarise
from .schemas import (
    AssessmentState,
    BaselineKind,
    BaselineSummary,
    BaselineWindow,
    FeatureBaseline,
    FeatureDeviation,
    InjectionShape,
    RiskAssessment,
)
from .store import AppendResult, Observation, TimeSeriesStore

__all__ = [
    "AppendResult",
    "AssessmentState",
    "BaselineEngine",
    "BaselineKind",
    "BaselineSummary",
    "BaselineWindow",
    "FeatureBaseline",
    "FeatureDeviation",
    "InjectionShape",
    "InjectionSpec",
    "Observation",
    "RiskAssessment",
    "TimeSeriesStore",
    "inject",
    "robust_z",
    "summarise",
]
