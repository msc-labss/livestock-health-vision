"""Synthetic anomaly injection, and the marking that follows it everywhere.

Injection happens at the time-series layer, not the pixel layer. The stub exists
to exercise baselines, scoring, thresholding and alerting, all of which consume
the series. Injecting at the pixel layer would additionally test perception's
response to synthetic imagery, which is a different question and one that is
easy to answer misleadingly.

Every artefact derived from injected data carries the marker. The marker is not
a convenience: without it an evaluation report cannot state which of its numbers
rest on observation and which on fabrication.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

from ..phenotype.schemas import FeatureRecord
from .schemas import InjectionShape

__all__ = ["InjectionSpec", "inject", "summarise"]


@dataclass(frozen=True)
class InjectionSpec:
    """A declared deviation: which feature, how large, what shape, and when."""

    feature: str
    magnitude: float
    shape: InjectionShape
    starts_at: datetime
    ends_at: datetime | None = None
    animal_id: str = ""
    note: str = ""

    def weight(self, at: datetime) -> float:
        """The share of the magnitude applying at ``at``, per the declared shape."""
        if at < self.starts_at:
            return 0.0
        if self.shape is InjectionShape.STEP:
            return 1.0 if self.ends_at is None or at <= self.ends_at else 0.0
        if self.shape is InjectionShape.SPIKE:
            return 1.0 if self.ends_at is None or at <= self.ends_at else 0.0

        if self.ends_at is None:
            return 1.0
        span = (self.ends_at - self.starts_at).total_seconds()
        if span <= 0:
            return 1.0 if at <= self.ends_at else 0.0
        position = (at - self.starts_at).total_seconds() / span
        if position > 1.0:
            return 0.0 if self.shape is InjectionShape.TRANSIENT else 1.0
        if self.shape is InjectionShape.RAMP:
            return position
        # Transient: rises and falls again within the span.
        return 1.0 - abs(2.0 * position - 1.0)


def inject(
    records: Iterable[FeatureRecord],
    specs: Iterable[InjectionSpec],
) -> list[FeatureRecord]:
    """Apply declared deviations, marking every record they touch.

    The magnitude is expressed in the feature's own unit and added to the value,
    so a report can state exactly how large the injected deviation was.
    """
    specs = list(specs)
    output: list[FeatureRecord] = []

    for record in records:
        applicable = [
            spec
            for spec in specs
            if (not spec.animal_id or spec.animal_id == record.animal_id)
            and record.observed_at is not None
        ]
        offsets: dict[str, float] = {}
        for spec in applicable:
            weight = spec.weight(record.observed_at)
            if weight:
                offsets[spec.feature] = offsets.get(spec.feature, 0.0) + spec.magnitude * weight

        if not offsets:
            output.append(record)
            continue

        features = tuple(
            dataclasses.replace(
                feature,
                value=feature.value + offsets[feature.name],
                note=(
                    f"{feature.note + '; ' if feature.note else ''}"
                    f"includes an injected offset of {offsets[feature.name]:+g} {feature.unit}"
                ),
            )
            if feature.name in offsets
            else feature
            for feature in record.features
        )
        output.append(dataclasses.replace(record, features=features, injected=True))

    return output


def summarise(specs: Iterable[InjectionSpec]) -> list[str]:
    """One line per declared injection, for a report to carry verbatim."""
    lines = []
    for spec in specs:
        window = spec.starts_at.isoformat()
        if spec.ends_at is not None:
            window += f" to {spec.ends_at.isoformat()}"
        subject = spec.animal_id or "every animal"
        lines.append(
            f"{spec.feature}: {spec.magnitude:+g} as a {spec.shape} on {subject} from {window}"
        )
    return lines
