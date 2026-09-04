"""The pipeline: registration through to exported events, in one invocation.

Every stage writes durable records before the next reads them, so any stage can
be re-run on its own. That is not a convenience — it is the property the P0 gate
turns on, and the cut line P3 needs when perception moves to the edge and
inference stays central.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .baseline.baselines import BaselineEngine
from .baseline.injection import InjectionSpec, inject, summarise
from .baseline.schemas import RiskAssessment
from .baseline.store import TimeSeriesStore
from .config import ResolvedConfig
from .events.build import EventBuilder
from .events.clips import ClipRetainer
from .events.export import EventExporter, FileExportAdapter
from .events.masking import Masker
from .events.policy import ThresholdPolicy
from .events.schemas import EventLevel, HealthEvent, ObservationWindow
from .identity.anchor import AnchorSource
from .identity.reid import ReferenceGallery
from .identity.resolve import IdentityResolver
from .identity.schemas import IdentityAssignment
from .ingest.source import RegisteredSource
from .ingest.stream import Ingestor
from .perception.detect import Detector, IntensityBlobDetector, UltralyticsDetector
from .perception.pose import PoseBackend, PoseEstimator
from .perception.report import PerceptionReport
from .perception.schemas import Pose, Tracklet
from .perception.track import Tracker
from .phenotype.features import FeatureExtractor
from .phenotype.passes import segment_passes
from .phenotype.schemas import FeatureRecord, LanePass
from .profiles import SpeciesProfile
from .stagestore import StageStore

__all__ = ["PipelineResult", "Pipeline"]

DETECTIONS = "detections"
TRACKLETS = "tracklets"
POSES = "poses"
ASSIGNMENTS = "identity_assignments"
PASSES = "passes"
FEATURES = "features"
ASSESSMENTS = "assessments"
EVENTS = "events"


@dataclass
class PipelineResult:
    config_digest: str = ""
    sources: int = 0
    perception: list[PerceptionReport] = field(default_factory=list)
    identity: str = ""
    passes: int = 0
    valid_passes: int = 0
    observations: int = 0
    unattributed: int = 0
    audited: int = 0
    assessments: int = 0
    scored: int = 0
    insufficient_history: int = 0
    events: int = 0
    alerts: int = 0
    clips_retained: int = 0
    clip_failures: int = 0
    exported: int = 0
    undelivered: int = 0
    injections: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"run {self.config_digest[:12]}: {self.sources} source(s)"]
        for report in self.perception:
            lines.append("  " + report.describe().replace("\n", "\n  "))
        if self.identity:
            lines.append(f"  {self.identity}")
        lines.append(
            f"  passes: {self.passes} ({self.valid_passes} valid); "
            f"series: {self.observations} appended, {self.unattributed} unattributed, "
            f"{self.audited} retained for audit"
        )
        lines.append(
            f"  assessments: {self.assessments} "
            f"({self.scored} scored, {self.insufficient_history} insufficient history)"
        )
        lines.append(
            f"  events: {self.events} ({self.alerts} alert-level); "
            f"clips: {self.clips_retained} retained, {self.clip_failures} not retained"
        )
        lines.append(f"  export: {self.exported} delivered, {self.undelivered} undelivered")
        for injection in self.injections:
            lines.append(f"  injected: {injection}")
        return "\n".join(lines)


class Pipeline:
    """Runs the stages, each reading and writing its own durable records."""

    def __init__(
        self,
        config: ResolvedConfig,
        profile: SpeciesProfile,
        output_root: str | Path,
        *,
        detector_backend=None,
        pose_backend: PoseBackend | None = None,
        anchor_source: AnchorSource | None = None,
        gallery: ReferenceGallery | None = None,
        masker: Masker | None = None,
        weights_root: str | Path | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.profile = profile
        self.output_root = Path(output_root)
        self.store = StageStore(self.output_root)
        self.series = TimeSeriesStore(self.output_root / "series", profile.feature_set)
        self.detector_backend = detector_backend or self._build_detector(weights_root)
        self.pose_backend = pose_backend or self._build_pose(weights_root)
        self.anchor_source = anchor_source
        self.gallery = gallery
        self.masker = masker
        self.now = now or datetime.now(UTC)
        self.result = PipelineResult(config_digest=config.digest)

    # -- backends -----------------------------------------------------------

    def _build_detector(self, weights_root):
        if self.config.perception.detector_backend == "intensity-blob":
            return IntensityBlobDetector()
        reference = self.profile.weight("detector")
        weights = str(Path(weights_root or "weights") / Path(reference.uri).name)
        return UltralyticsDetector(
            weights,
            name=reference.name,
            version=reference.version,
            target_classes=reference.target_classes,
            device=self.config.perception.device,
            confidence_threshold=self.config.perception.detection_threshold,
        )

    def _build_pose(self, weights_root) -> PoseBackend | None:
        if self.config.perception.pose_backend == "none":
            return None
        from .perception.pose import pose_backend_from_profile

        reference = self.profile.weight("pose")
        weights = str(Path(weights_root or "weights") / Path(reference.uri).name)
        return pose_backend_from_profile(self.profile, self.config, weights_path=weights)

    # -- stage 1: ingest, detection, tracking, pose -------------------------

    def run_perception(self, sources: Sequence[RegisteredSource]) -> PipelineResult:
        for stage in (DETECTIONS, TRACKLETS, POSES):
            self.store.clear(stage)

        for source in sources:
            self.result.sources += 1
            detector = Detector(self.detector_backend, self.config)
            estimator = (
                PoseEstimator(self.pose_backend, self.profile, self.config)
                if self.pose_backend is not None
                else None
            )

            frames = list(Ingestor(source, self.config).iter_frames())
            images = {frame.index: frame.image for frame in frames}
            records = detector.detect_all(frames)

            tracker = Tracker(self.config, source_id=source.source_id)
            tracklets = tracker.track(records)

            poses: list[Pose] = []
            if estimator is not None:
                by_tracklet = {
                    detection.detection_id: tracklet.tracklet_id
                    for tracklet in tracklets
                    for detection in tracklet.detections
                }
                for record in records:
                    for detection in record.detections:
                        image = images.get(detection.frame_index)
                        if image is None:
                            continue
                        poses.append(
                            estimator.estimate(
                                image,
                                detection,
                                tracklet_id=by_tracklet.get(detection.detection_id, ""),
                            )
                        )

            self.store.write(
                DETECTIONS,
                records,
                site_of=lambda r: r.provenance.site_key,
                day_of=lambda r: r.provenance.day_key,
                source_of=lambda r: r.provenance.source_id,
                sort_of=lambda r: r.provenance.frame_index,
            )
            self.store.write(TRACKLETS, tracklets, sort_of=lambda r: r.first_frame_index)
            self.store.write(
                POSES,
                poses,
                site_of=lambda r: r.provenance.site_key,
                day_of=lambda r: r.provenance.day_key,
                source_of=lambda r: r.provenance.source_id,
                sort_of=lambda r: r.provenance.frame_index,
            )

            self.result.perception.append(
                PerceptionReport(
                    source_id=source.source_id,
                    detector_identity=detector.model_identity,
                    pose_identity=estimator.model_identity if estimator else "none",
                    skeleton=(
                        f"{self.profile.skeleton.identifier}@{self.profile.skeleton.version}"
                    ),
                    frames_processed=detector.frames_processed,
                    frames_without_detection=detector.frames_without_detection,
                    detections_emitted=detector.detections_emitted,
                    detections_low_confidence=detector.low_confidence_detections,
                    tracklets_formed=tracker.report.tracklets_formed,
                    terminations=dict(tracker.report.terminations),
                    poses_emitted=estimator.poses_emitted if estimator else 0,
                    poses_low_confidence=estimator.low_confidence_poses if estimator else 0,
                    keypoints_emitted=estimator.keypoints_emitted if estimator else 0,
                    keypoints_not_visible=estimator.keypoints_not_visible if estimator else 0,
                )
            )
        return self.result

    # -- stage 2: identity --------------------------------------------------

    def run_identity(self, *, crops: dict | None = None) -> list[IdentityAssignment]:
        self.store.clear(ASSIGNMENTS)
        tracklets = self.store.read(TRACKLETS, Tracklet)
        resolver = IdentityResolver(
            self.config,
            anchor_source=self.anchor_source,
            gallery=self.gallery,
            now=self.now,
        )
        assignments = resolver.resolve_all(tracklets, crops=crops)
        self.store.write(ASSIGNMENTS, assignments)
        self.result.identity = resolver.report.describe()
        return assignments

    # -- stage 3: phenotype -------------------------------------------------

    def run_phenotype(self, *, frame_width: int, frame_height: int) -> list[FeatureRecord]:
        self.store.clear(PASSES)
        self.store.clear(FEATURES)

        tracklets = self.store.read(TRACKLETS, Tracklet)
        poses = self.store.read(POSES, Pose)
        assignments = {a.tracklet_id: a for a in self.store.read(ASSIGNMENTS, IdentityAssignment)}
        poses_by_tracklet: dict[str, list[Pose]] = {}
        for pose in poses:
            poses_by_tracklet.setdefault(pose.tracklet_id, []).append(pose)

        extractor = FeatureExtractor(self.profile, self.config)
        all_passes: list[LanePass] = []
        records: list[FeatureRecord] = []

        for tracklet in tracklets:
            lane_passes = segment_passes(
                tracklet, self.config, frame_width=frame_width, frame_height=frame_height
            )
            all_passes.extend(lane_passes)
            assignment = assignments.get(tracklet.tracklet_id)
            for lane_pass in lane_passes:
                records.append(
                    extractor.extract(
                        lane_pass,
                        poses_by_tracklet.get(tracklet.tracklet_id, []),
                        animal_id=assignment.animal_id if assignment else "",
                        identity_resolved=bool(assignment and assignment.resolved),
                    )
                )

        self.store.write(PASSES, all_passes, sort_of=lambda r: r.first_frame_index)
        self.store.write(FEATURES, records)
        self.result.passes = len(all_passes)
        self.result.valid_passes = sum(1 for r in records if r.valid)
        return records

    # -- stage 4: baseline --------------------------------------------------

    def run_baseline(self, *, injections: Iterable[InjectionSpec] = ()) -> list[RiskAssessment]:
        """Rebuild the series and score it. Reads feature records, not video."""
        import shutil

        self.store.clear(ASSESSMENTS)
        if self.series.root.exists():
            shutil.rmtree(self.series.root)
        self.series.root.mkdir(parents=True, exist_ok=True)

        records = self.store.read(FEATURES, FeatureRecord)
        injections = list(injections)
        if injections:
            records = inject(records, injections)
            self.result.injections = summarise(injections)
            self.store.clear(FEATURES)
            self.store.write(FEATURES, records)

        appended = self.series.append(records)
        self.result.observations = appended.appended
        self.result.unattributed = appended.unattributed
        self.result.audited = appended.audited

        engine = BaselineEngine(self.series, self.profile, self.config)
        assessments = [engine.assess(o) for o in self.series.read()]
        self.store.write(ASSESSMENTS, assessments, sort_of=lambda r: 0)

        self.result.assessments = len(assessments)
        self.result.scored = sum(1 for a in assessments if a.scored)
        self.result.insufficient_history = sum(1 for a in assessments if not a.scored)
        return assessments

    # -- stage 5: events ----------------------------------------------------

    def run_events(self, *, sources: Sequence[RegisteredSource] = ()) -> list[HealthEvent]:
        """Build, retain evidence for, and export events. Reads assessments."""
        self.store.clear(EVENTS)

        assessments = self.store.read(ASSESSMENTS, RiskAssessment)
        records = {r.pass_id: r for r in self.store.read(FEATURES, FeatureRecord)}
        lane_passes = {p.pass_id: p for p in self.store.read(PASSES, LanePass)}
        by_source = {s.source_id: s for s in sources}

        policy = ThresholdPolicy.from_config(self.config)
        builder = EventBuilder(
            self.config,
            policy,
            model_identities={
                "detector": self.detector_backend.model_identity,
                "pose": (
                    self.pose_backend.model_identity if self.pose_backend is not None else "none"
                ),
                "skeleton": (f"{self.profile.skeleton.identifier}@{self.profile.skeleton.version}"),
            },
        )
        retainer = ClipRetainer(self.output_root / "evidence", self.config, masker=self.masker)

        events: list[HealthEvent] = []
        for assessment in assessments:
            record = records.get(assessment.pass_id)
            lane_pass = lane_passes.get(assessment.pass_id)
            if record is None or lane_pass is None:
                continue
            window = ObservationWindow(
                start=lane_pass.first_timestamp or assessment.observed_at,
                end=lane_pass.last_timestamp or assessment.observed_at,
            )
            level = policy.level_for(assessment)
            clip = None
            if level is EventLevel.ALERT:
                source = by_source.get(lane_pass.source_id)
                if source is not None:
                    clip = retainer.retain(
                        level=level,
                        event_id=f"{assessment.assessment_id}:event",
                        media_path=source.media_path,
                        media_kind=source.kind,
                        first_frame_index=lane_pass.first_frame_index,
                        last_frame_index=lane_pass.last_frame_index,
                        frame_rate=self._frame_rate(lane_pass),
                    )
            events.append(builder.build(assessment, record, window=window, evidence_clip=clip))

        self.store.write(EVENTS, events, sort_of=lambda r: 0)

        adapter = FileExportAdapter(self.output_root / "export")
        exporter = EventExporter(adapter, retention_dir=self.output_root / "export" / "pending")
        report = exporter.export(events)

        self.result.events = len(events)
        self.result.alerts = sum(1 for e in events if e.is_alert)
        self.result.clips_retained = retainer.clips_written
        self.result.clip_failures = retainer.retention_failures
        self.result.exported = report.delivered
        self.result.undelivered = report.undelivered
        return events

    def _frame_rate(self, lane_pass: LanePass) -> float:
        if lane_pass.first_timestamp is None or lane_pass.last_timestamp is None:
            return 0.0
        span = (lane_pass.last_timestamp - lane_pass.first_timestamp).total_seconds()
        frames = lane_pass.last_frame_index - lane_pass.first_frame_index
        return frames / span if span > 0 and frames > 0 else 0.0

    # -- everything ---------------------------------------------------------

    def run(
        self,
        sources: Sequence[RegisteredSource],
        *,
        frame_width: int,
        frame_height: int,
        injections: Iterable[InjectionSpec] = (),
        crops: dict | None = None,
    ) -> PipelineResult:
        self.run_perception(sources)
        self.run_identity(crops=crops)
        self.run_phenotype(frame_width=frame_width, frame_height=frame_height)
        self.run_baseline(injections=injections)
        self.run_events(sources=sources)
        return self.result
