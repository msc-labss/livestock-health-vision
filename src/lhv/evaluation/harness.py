"""Assembling a report from a run's materialised records.

The harness reads what the stages wrote rather than re-running them, which is
why the stage boundaries are durable in the first place. It refuses a leaking
split before computing anything, keeps the metric families apart, and attaches
the declared limitations to every report it produces.

A family with no ground truth to measure against is absent from the report and
said to be absent. It is never reported as zero, and never quietly omitted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..baseline.store import TimeSeriesStore
from ..config import ResolvedConfig
from ..events.schemas import HealthEvent
from ..identity.schemas import IdentityAssignment
from ..perception.schemas import Pose, Tracklet
from ..phenotype.schemas import FeatureRecord
from ..profiles import SpeciesProfile
from ..stagestore import StageStore
from .metrics import (
    LabelledBox,
    LabelledKeypoint,
    detection_metrics,
    identity_metrics,
    operational_metrics,
    phenotype_metrics,
    pose_metrics,
    tracking_metrics,
)
from .report import EvaluationReport, ReportInputs
from .splits import SplitItem, SplitKind, assert_no_leakage, build_split

__all__ = ["PerceptionLabels", "EvaluationHarness"]


@dataclass
class PerceptionLabels:
    """Ground truth a perception family can be measured against."""

    boxes: Sequence[LabelledBox] = ()
    keypoints: Sequence[LabelledKeypoint] = ()
    identity_by_tracklet: dict[str, str] | None = None
    keypoint_normaliser: float = 100.0


class EvaluationHarness:
    def __init__(
        self,
        store: StageStore,
        series: TimeSeriesStore,
        config: ResolvedConfig,
        profile: SpeciesProfile,
    ) -> None:
        self.store = store
        self.series = series
        self.config = config
        self.profile = profile

    # -- splits -------------------------------------------------------------

    def split_items(self) -> list[SplitItem]:
        """Evaluable units, keyed from the provenance the records carry."""
        items = []
        for record in self.store.read("features", FeatureRecord):
            items.append(
                SplitItem(
                    item_id=record.pass_id,
                    animal_key=record.animal_id,
                    day_key=record.day_key,
                    site_key=record.site_key,
                )
            )
        return items

    # -- report -------------------------------------------------------------

    def build(
        self,
        *,
        split_kind: SplitKind = SplitKind.ANIMAL,
        labels: PerceptionLabels | None = None,
        reference_event_name: str = "none available in P0",
        reference_times: dict | None = None,
        injections: Sequence[str] = (),
        model_identities: dict[str, str] | None = None,
        dataset_name: str | None = None,
        dataset_version: str | None = None,
    ) -> EvaluationReport:
        items = [i for i in self.split_items() if i.animal_key]
        split = build_split(
            items,
            split_kind,
            test_fraction=self.config.evaluation.test_fraction,
            seed=self.config.seed,
        )
        # Refused before a single metric is computed.
        assert_no_leakage(split, items)

        observations = self.series.read()
        site_keys = tuple(sorted({o.site_key for o in observations} or {"unknown"}))

        report = EvaluationReport(
            inputs=ReportInputs(
                dataset_name=dataset_name or self.config.dataset_name,
                dataset_version=dataset_version or self.config.dataset_version,
                model_identities=model_identities
                or {role: str(m) for role, m in self.config.models.items()},
                split=split,
                config_digest=self.config.digest,
                feature_set_version=self.profile.feature_set.version,
                skeleton=(f"{self.profile.skeleton.identifier}@{self.profile.skeleton.version}"),
                site_keys=site_keys,
                placeholder_weights={
                    w.role: w.placeholder_reason for w in self.profile.placeholder_weights()
                },
                unavailable_features={
                    f.name: " ".join(f.unavailable_reason.split())
                    for f in self.profile.feature_set.unavailable
                },
                feature_names=self.profile.feature_set.names,
            ),
            injections=list(injections),
        )

        test_animals = set(split.test_keys) if split_kind is SplitKind.ANIMAL else None
        self._add_perception(report, labels, test_animals)
        self._add_phenotype(report, observations, test_animals)
        self._add_identity(report, labels)
        self._add_operational(report, observations, reference_event_name, reference_times)
        return report

    # -- families -----------------------------------------------------------

    def _add_perception(
        self,
        report: EvaluationReport,
        labels: PerceptionLabels | None,
        test_animals: set[str] | None,
    ) -> None:
        if labels is None or not labels.boxes:
            report.limitations.append(
                "Perception metrics were not computed: no labelled ground truth was supplied "
                "for this run. Detection, tracking and pose accuracy are therefore unmeasured "
                "rather than measured as zero."
            )
            return

        tracklets = self.store.read("tracklets", Tracklet)
        track_of_pose: dict[str, str] = {}
        boxes = list(labels.boxes)
        keypoints = list(labels.keypoints)
        poses = self.store.read("poses", Pose)

        # Score the held-out partition only. Measuring perception over animals
        # the split placed in training would report a number that no split
        # protects.
        if test_animals:
            assigned = {
                a.tracklet_id: a.animal_id
                for a in self.store.read("identity_assignments", IdentityAssignment)
                if a.resolved
            }
            held_out_tracklets = [
                t for t in tracklets if assigned.get(t.tracklet_id) in test_animals
            ]
            if held_out_tracklets:
                tracklets = held_out_tracklets
                held_out_ids = {t.tracklet_id for t in tracklets}
                # Labels are filtered by animal, not merely by frame. Keeping
                # every label in a frame a held-out animal appears in would
                # count its co-present neighbours — animals the split put in
                # training — as detections this run missed.
                boxes = [b for b in boxes if b.instance in test_animals]
                keypoints = [k for k in keypoints if k.instance in test_animals]
                poses = [p for p in poses if p.tracklet_id in held_out_ids]
                track_of_pose = {p.pose_id: assigned.get(p.tracklet_id, "") for p in poses}
                report.limitations.append(
                    f"Perception metrics cover the held-out partition only: "
                    f"{len(tracklets)} tracklet(s) over {len(test_animals)} animal(s)."
                )

        detections = [d for t in tracklets for d in t.detections]

        # A family measured against a backend that serves the labels is not a
        # measurement of that backend. Saying so is the difference between a
        # report and a decoration.
        served = {
            "detection": {d.model_identity for d in detections},
            "pose": {p.model_identity for p in poses},
        }
        for family, identities in served.items():
            if any("label" in identity for identity in identities):
                report.limitations.append(
                    f"The {family} family was produced by a backend serving the dataset's own "
                    f"labels ({', '.join(sorted(identities))}), so its scores are near-perfect "
                    f"by construction and measure nothing about a model. They are reported "
                    f"because the stages downstream consumed exactly these values; the tracking "
                    f"family, which was not handed its answer, is the one that measures this "
                    f"system."
                )

        report.add(
            detection_metrics(
                detections,
                boxes,
                iou_threshold=self.config.evaluation.detection_iou_threshold,
            )
        )
        report.add(
            tracking_metrics(
                tracklets,
                boxes,
                iou_threshold=self.config.evaluation.detection_iou_threshold,
            )
        )
        if keypoints:
            report.add(
                pose_metrics(
                    poses,
                    keypoints,
                    distance_threshold=self.config.evaluation.keypoint_distance_threshold,
                    normaliser=labels.keypoint_normaliser,
                    track_of_pose=track_of_pose,
                )
            )

    def _add_phenotype(
        self, report: EvaluationReport, observations, test_animals: set[str] | None
    ) -> None:
        held_out = (
            [o for o in observations if o.animal_id in test_animals]
            if test_animals
            else list(observations)
        )
        report.add(phenotype_metrics(held_out, self.profile.feature_set.names))

    def _add_identity(self, report: EvaluationReport, labels: PerceptionLabels | None) -> None:
        if labels is None or not labels.identity_by_tracklet:
            report.limitations.append(
                "Identity accuracy was not computed: no labelled identity was supplied. The "
                "anchored path and the visual fallback are therefore unmeasured."
            )
            return
        assignments = self.store.read("identity_assignments", IdentityAssignment)
        for method in ("external_anchor", "visual_fallback"):
            report.add(identity_metrics(assignments, labels.identity_by_tracklet, method=method))
        report.limitations.append(
            "The anchored path in P0 is the dataset's own ground-truth identity routed through "
            "the anchor interface, so its accuracy is perfect by construction and says nothing "
            "about a real identifier stream. The visual fallback is reported separately above "
            "for that reason."
        )

    def _add_operational(
        self,
        report: EvaluationReport,
        observations,
        reference_event_name: str,
        reference_times: dict | None,
    ) -> None:
        events = self.store.read("events", HealthEvent)
        animal_days = len({(o.animal_id, o.day_key) for o in observations})
        policy = (
            f"{self.config.events.threshold_policy_name}@"
            f"{self.config.events.threshold_policy_version}"
        )
        report.add(
            operational_metrics(
                events,
                animal_days=animal_days,
                policy_identity=policy,
                reference_event_name=reference_event_name,
                reference_times=reference_times,
            )
        )
        if not reference_times:
            report.limitations.append(
                "Lead time is unmeasured: P0 has no clinical reference event to measure it "
                "against. It becomes measurable in P1, against parallel human locomotion "
                "scoring."
            )
