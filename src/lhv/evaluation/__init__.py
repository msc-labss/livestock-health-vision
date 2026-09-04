"""Cross-cutting: split construction, separated metric families, reports.

Attaches to the materialised records of every stage rather than sitting inside
any one of them.
"""

from .harness import EvaluationHarness, PerceptionLabels
from .metrics import (
    LabelledBox,
    LabelledKeypoint,
    MetricFamily,
    detection_metrics,
    identity_metrics,
    operational_metrics,
    phenotype_metrics,
    pose_metrics,
    tracking_metrics,
)
from .report import EvaluationReport, ReportInputs
from .splits import SplitDefinition, SplitItem, SplitKind, assert_no_leakage, build_split

__all__ = [
    "EvaluationHarness",
    "EvaluationReport",
    "LabelledBox",
    "LabelledKeypoint",
    "MetricFamily",
    "PerceptionLabels",
    "ReportInputs",
    "SplitDefinition",
    "SplitItem",
    "SplitKind",
    "assert_no_leakage",
    "build_split",
    "detection_metrics",
    "identity_metrics",
    "operational_metrics",
    "phenotype_metrics",
    "pose_metrics",
    "tracking_metrics",
]
