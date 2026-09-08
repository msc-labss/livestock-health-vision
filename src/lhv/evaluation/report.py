"""The evaluation report.

A report records what produced it — dataset version, model identities, split
definition, configuration digest — so that regenerating it from those recorded
inputs is a check anyone can run rather than a claim they have to take.

It also carries its limitations. In P0 that means stating on every report that
health inference is stubbed and that the results are not clinical evidence, and
stating the number of distinct sites, with a note when site-disjoint validation
did not run. A limitation that lives only in a commit message is a limitation
nobody downstream will ever see.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime

from ..schema import canonical_json
from .metrics import MetricFamily
from .splits import SplitDefinition

__all__ = ["EventLevelName", "ReportInputs", "EvaluationReport"]


class EventLevelName:
    ALERT = "alert"


@dataclass(frozen=True)
class ReportInputs:
    """Everything a report is a function of. Recorded so it can be re-run."""

    dataset_name: str
    dataset_version: str
    model_identities: dict[str, str]
    split: SplitDefinition
    config_digest: str
    feature_set_version: str
    skeleton: str
    site_keys: tuple[str, ...]
    # role -> reason, for weights the profile declares a placeholder. Part of the
    # recorded inputs because two runs differing only in whether their weights
    # were placeholders are not the same evaluation.
    placeholder_weights: dict[str, str] = field(default_factory=dict)
    # feature name -> reason, for features the profile declares but the current
    # skeleton or backend cannot compute. Part of the recorded inputs because a
    # phenotype metric over three features means something different from the
    # same metric over nine.
    unavailable_features: dict[str, str] = field(default_factory=dict)
    # Every feature the profile declares, computable or not, so the report can
    # say "6 of 9" rather than only naming the six.
    feature_names: tuple[str, ...] = ()

    def fingerprint(self) -> str:
        return hashlib.sha256(
            canonical_json(
                {
                    "dataset_name": self.dataset_name,
                    "dataset_version": self.dataset_version,
                    "model_identities": self.model_identities,
                    "split": self.split.to_dict(),
                    "config_digest": self.config_digest,
                    "feature_set_version": self.feature_set_version,
                    "skeleton": self.skeleton,
                    "site_keys": sorted(self.site_keys),
                    "placeholder_weights": dict(sorted(self.placeholder_weights.items())),
                    "unavailable_features": dict(sorted(self.unavailable_features.items())),
                }
            ).encode("utf-8")
        ).hexdigest()


@dataclass
class EvaluationReport:
    """Separated metric families, recorded inputs, and declared limitations."""

    inputs: ReportInputs
    families: list[MetricFamily] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    injections: list[str] = field(default_factory=list)
    generated_at: datetime | None = None

    # -- construction -------------------------------------------------------

    def add(self, family: MetricFamily) -> EvaluationReport:
        self.families.append(family)
        return self

    def family(self, name: str) -> MetricFamily | None:
        for candidate in self.families:
            if candidate.name == name:
                return candidate
        return None

    @property
    def family_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.families)

    @property
    def site_count(self) -> int:
        return len(set(self.inputs.site_keys))

    def declared_limitations(self) -> list[str]:
        """The limitations that apply to every P0 report, plus any added ones."""
        stated = [
            "Health inference is stubbed. The baseline, risk and alerting layers were "
            "exercised against injected synthetic deviations, not fitted to clinical "
            "outcomes. Nothing in this report is clinical evidence.",
            f"Evaluation covers {self.site_count} distinct site(s).",
        ]
        if self.site_count < 2:
            stated.append(
                "Site-disjoint validation was not exercised: a site-disjoint split cannot "
                "be constructed from a single site. Domain shift between farms is therefore "
                "unmeasured."
            )
        unavailable = self.inputs.unavailable_features
        if unavailable:
            declared = len(self.inputs.feature_names) or len(unavailable)
            names = ", ".join(sorted(unavailable))
            stated.append(
                f"{len(unavailable)} of {declared} declared features could not be computed "
                f"under this configuration and carry no value: {names}. Phenotype metrics here "
                f"cover the remainder, which is a smaller claim than the feature set's size "
                f"suggests."
            )
            for name, reason in sorted(unavailable.items()):
                if reason:
                    stated.append(f"  {name} is unavailable because {reason}")

        for role, reason in sorted(self.inputs.placeholder_weights.items()):
            stated.append(
                f"The {role} weights are declared a placeholder"
                + (f" ({reason})" if reason else "")
                + ". Any metric family whose output came from them measures the "
                "placeholder rather than an achievable result."
            )
        if self.injections:
            stated.append(
                "Injected data was present in this evaluation. Every affected output is "
                "marked, and the injections are listed below."
            )
        return stated + list(self.limitations)

    # -- reproducibility ----------------------------------------------------

    def content(self) -> dict:
        """Everything the report asserts, excluding when it was generated."""
        return {
            "inputs": {
                "dataset_name": self.inputs.dataset_name,
                "dataset_version": self.inputs.dataset_version,
                "model_identities": dict(sorted(self.inputs.model_identities.items())),
                "split": self.inputs.split.to_dict(),
                "config_digest": self.inputs.config_digest,
                "feature_set_version": self.inputs.feature_set_version,
                "skeleton": self.inputs.skeleton,
                "site_keys": sorted(self.inputs.site_keys),
            },
            "families": [
                {
                    "name": family.name,
                    "metrics": {k: round(v, 10) for k, v in sorted(family.metrics.items())},
                    "counts": dict(sorted(family.counts.items())),
                    "notes": list(family.notes),
                }
                for family in self.families
            ],
            "limitations": self.declared_limitations(),
            "injections": list(self.injections),
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(canonical_json(self.content()).encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        return json.dumps(self.content(), indent=2, sort_keys=True)

    # -- rendering ----------------------------------------------------------

    def render(self) -> str:
        lines = [
            "# Evaluation report",
            "",
            "## Recorded inputs",
            f"- dataset: {self.inputs.dataset_name}@{self.inputs.dataset_version}",
            f"- configuration digest: {self.inputs.config_digest}",
            f"- feature set: v{self.inputs.feature_set_version}",
            f"- skeleton: {self.inputs.skeleton}",
            f"- split: {self.inputs.split.reference}",
            "- model identities:",
        ]
        for role, identity in sorted(self.inputs.model_identities.items()):
            lines.append(f"    - {role}: {identity}")
        lines.append(f"- report fingerprint: {self.fingerprint()}")
        lines.append("")

        for family in self.families:
            lines.append(f"## {family.name}")
            for key, value in sorted(family.metrics.items()):
                lines.append(f"- {key}: {value:.4f}")
            for key, count in sorted(family.counts.items()):
                lines.append(f"- {key}: {count}")
            for note in family.notes:
                lines.append(f"- _{note}_")
            lines.append("")

        lines.append("## Declared limitations")
        for limitation in self.declared_limitations():
            lines.append(f"- {limitation}")
        if self.injections:
            lines.append("")
            lines.append("## Injected deviations present in this evaluation")
            for injection in self.injections:
                lines.append(f"- {injection}")
        lines.append("")
        lines.append(
            "No aggregate score is reported across metric families. They answer different "
            "questions and fail for different reasons."
        )
        return "\n".join(lines)
