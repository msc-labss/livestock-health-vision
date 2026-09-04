"""End to end: registration through to exported events, and stage independence."""

from __future__ import annotations

import dataclasses
import json
from datetime import timedelta
from pathlib import Path

import pytest

from lhv.baseline import InjectionShape, InjectionSpec
from lhv.config import ModelIdentity, ResolvedConfig
from lhv.datasets import AccessTerms, DatasetRegistration
from lhv.events import EventLevel, RegionMasker
from lhv.ingest import register_source
from lhv.pipeline import EVENTS, FEATURES, Pipeline
from lhv.profiles import load_profile

from .conftest import (
    FIXED_START,
    LabelPoseBackend,
    anchors_from_labels,
    write_labelled_lane_dataset,
)

SITE = "site-synth"
FRAME_WIDTH, FRAME_HEIGHT = 320, 240


@pytest.fixture(scope="module")
def profile():
    return load_profile("cattle")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory, profile):
    root = tmp_path_factory.mktemp("labelled-dataset")
    write_labelled_lane_dataset(root, profile, animals=4, days=8, frames=40)
    return root


@pytest.fixture
def labels(dataset):
    return json.loads((dataset / "labels.json").read_text(encoding="utf-8"))


@pytest.fixture
def registration():
    return DatasetRegistration(
        name="synthetic-lane",
        version="1",
        site_key=SITE,
        camera_ids=("cam-lane-1",),
        animal_set_keys=("synth-herd",),
        access=AccessTerms(licence="test-only", access_route="open-download"),
    )


@pytest.fixture
def pipeline_config(profile):
    return ResolvedConfig(
        species_profile=profile.species,
        species_profile_version=profile.version,
        dataset_name="synthetic-lane",
        dataset_version="1",
        models={
            "detector": ModelIdentity(name="intensity-blob", version="1", task="detect"),
            "pose": ModelIdentity(name="dataset-keypoint-label", version="1", task="pose"),
        },
        perception=dataclasses.replace(
            ResolvedConfig(
                species_profile="x",
                species_profile_version="0",
                dataset_name="x",
                dataset_version="1",
            ).perception,
            detector_backend="intensity-blob",
            pose_backend="label",
            detection_threshold=0.05,
            detection_low_confidence_threshold=0.10,
        ),
    )


def _sources(dataset, labels):
    sources = []
    for relative, entry in sorted(labels.items()):
        day_index = (
            __import__("datetime").date.fromisoformat(entry["day_key"]) - FIXED_START.date()
        ).days
        sources.append(
            register_source(
                source_id=f"synthetic-lane/{relative}",
                camera_id="cam-lane-1",
                site_key=SITE,
                animal_set_key="synth-herd",
                media_path=str(dataset / relative),
                dataset_name="synthetic-lane",
                dataset_version="1",
                start_timestamp=FIXED_START
                + timedelta(days=day_index, seconds=entry["start_offset_seconds"]),
                kind="video",
            )
        )
    return sources


def _pipeline(output, pipeline_config, profile, labels, **kwargs):
    return Pipeline(
        pipeline_config,
        profile,
        output,
        detector_backend=__import__(
            "lhv.perception", fromlist=["IntensityBlobDetector"]
        ).IntensityBlobDetector(threshold=150, min_area=300),
        pose_backend=LabelPoseBackend(labels),
        anchor_source=anchors_from_labels(labels, site_key=SITE, dataset_name="synthetic-lane"),
        masker=RegionMasker([(0.0, 0.0, 1.0, 0.15)]),
        now=FIXED_START,
        **kwargs,
    )


@pytest.fixture
def run(tmp_path, pipeline_config, profile, dataset, labels):
    pipeline = _pipeline(tmp_path / "run", pipeline_config, profile, labels)
    sources = _sources(dataset, labels)
    result = pipeline.run(sources, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    return pipeline, sources, result


# -- 10.1 the pipeline runs end to end from one invocation ------------------


def test_the_pipeline_runs_from_registration_to_exported_events(run) -> None:
    pipeline, sources, result = run

    assert result.sources == len(sources) == 32
    assert result.perception, "perception produced no report"
    assert sum(r.detections_emitted for r in result.perception) > 0
    assert sum(r.tracklets_formed for r in result.perception) > 0
    assert sum(r.poses_emitted for r in result.perception) > 0
    assert result.passes > 0
    assert result.valid_passes > 0
    assert result.observations > 0
    assert result.assessments > 0
    assert result.events > 0
    assert result.exported == result.events
    assert result.undelivered == 0


def test_no_manual_step_sits_between_the_stages(run) -> None:
    """Every stage's records were written by the run itself."""
    pipeline, _, _ = run
    assert set(pipeline.store.stages()) >= {
        "detections",
        "tracklets",
        "poses",
        "identity_assignments",
        "passes",
        "features",
        "assessments",
        "events",
    }


def test_identity_resolves_through_the_anchor_path(run) -> None:
    _, _, result = run
    assert "anchored" in result.identity
    assert result.observations > 0, "anchored passes must reach the time series"


def test_exported_events_are_readable_from_disk(run) -> None:
    pipeline, _, result = run
    exported = sorted((pipeline.output_root / "export").glob("*.json"))
    assert len(exported) == result.events

    payload = json.loads(exported[0].read_text(encoding="utf-8"))
    assert payload["schema_name"] == "health_event"
    assert payload["schema_version"]
    assert payload["stub_derived"] is True
    assert payload["non_clinical"] is True
    assert payload["threshold_policy"]["name"]


def test_the_run_records_its_configuration_digest(run, pipeline_config) -> None:
    _, _, result = run
    assert result.config_digest == pipeline_config.digest


def test_the_cli_runs_the_same_pipeline(tmp_path, dataset, labels, registration) -> None:
    """The single invocation the gate names is a real command."""
    from lhv.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--dataset",
            "cattleeyeview",
            "--data-root",
            str(tmp_path / "absent"),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert args.command == "run"
    assert args.stages == "all"


# -- 10.2 the identity fallback and a multi-day series ----------------------


def test_a_multi_day_series_populates(run) -> None:
    pipeline, _, _ = run
    days = {o.day_key for o in pipeline.series.read()}
    assert len(days) >= 5, "a longitudinal baseline needs more than one day"


def test_the_visual_fallback_path_resolves_when_no_anchor_exists(
    tmp_path, pipeline_config, profile, dataset, labels
) -> None:
    """With the anchor withheld, identity falls back to appearance and says so."""
    from lhv.identity import (
        AssignmentMethod,
        ColourHistogramEmbedding,
        IdentityAssignment,
        ReferenceGallery,
    )
    from lhv.perception import Tracklet

    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    import numpy as np

    for index, animal in enumerate(sorted({e["animal_id"] for e in labels.values()})):
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        patch[:, :] = (20 + 60 * index, 200 - 40 * index, 30 + 20 * index)
        gallery.enrol_image(animal, patch)

    pipeline = Pipeline(
        pipeline_config,
        profile,
        tmp_path / "fallback",
        detector_backend=__import__(
            "lhv.perception", fromlist=["IntensityBlobDetector"]
        ).IntensityBlobDetector(threshold=150, min_area=300),
        pose_backend=LabelPoseBackend(labels),
        anchor_source=None,
        gallery=gallery,
        now=FIXED_START,
    )
    sources = _sources(dataset, labels)[:4]
    pipeline.run_perception(sources)

    tracklets = pipeline.store.read("tracklets", Tracklet)
    assert tracklets
    crops = {}
    for index, tracklet in enumerate(tracklets):
        patch = np.zeros((32, 32, 3), dtype=np.uint8)
        patch[:, :] = (20 + 60 * (index % 4), 200 - 40 * (index % 4), 30 + 20 * (index % 4))
        crops[tracklet.tracklet_id] = patch

    assignments = pipeline.run_identity(crops=crops)
    methods = {a.method for a in assignments}
    assert AssignmentMethod.VISUAL_FALLBACK in methods
    for assignment in assignments:
        if assignment.method is AssignmentMethod.VISUAL_FALLBACK:
            assert assignment.confidence > 0
            assert assignment.evidence_reference.startswith("visual:")
    assert all(isinstance(a, IdentityAssignment) for a in assignments)


# -- 10.3 stages are independently re-runnable ------------------------------


def test_the_baseline_and_event_stages_re_run_without_recomputing_detection(run) -> None:
    pipeline, sources, first = run

    detections_before = sorted(
        p.stat().st_mtime_ns for p in pipeline.store.stage_dir("detections").rglob("*.parquet")
    )
    events_before = _event_fingerprints(pipeline)

    # Only the cheap stages, reading the records the expensive one wrote.
    fresh = Pipeline(
        pipeline.config,
        pipeline.profile,
        pipeline.output_root,
        detector_backend=pipeline.detector_backend,
        pose_backend=pipeline.pose_backend,
        anchor_source=pipeline.anchor_source,
        masker=pipeline.masker,
        now=FIXED_START,
    )
    fresh.run_baseline()
    fresh.run_events(sources=sources)

    detections_after = sorted(
        p.stat().st_mtime_ns for p in pipeline.store.stage_dir("detections").rglob("*.parquet")
    )
    assert detections_after == detections_before, "detection records must not be rewritten"
    assert _event_fingerprints(fresh) == events_before
    assert fresh.result.observations == first.observations
    assert fresh.result.events == first.events


def test_a_full_rerun_reproduces_the_same_events(
    tmp_path, pipeline_config, profile, dataset, labels, run
) -> None:
    pipeline, sources, _ = run
    other = _pipeline(tmp_path / "rerun", pipeline_config, profile, labels)
    other.run(sources, frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    assert _event_fingerprints(other) == _event_fingerprints(pipeline)


def _event_fingerprints(pipeline) -> list[tuple]:
    from lhv.events import HealthEvent

    return sorted(
        (
            event.animal_id,
            event.event_timestamp.isoformat(),
            str(event.level),
            None if event.risk_score is None else round(event.risk_score, 9),
            event.idempotency_key,
        )
        for event in pipeline.store.read(EVENTS, HealthEvent)
    )


# -- injection reaches the exported event -----------------------------------


def test_an_injected_deviation_reaches_an_exported_alert(
    tmp_path, pipeline_config, profile, dataset, labels
) -> None:
    from lhv.events import HealthEvent
    from lhv.phenotype import FeatureRecord

    pipeline = _pipeline(tmp_path / "injected", pipeline_config, profile, labels)
    sources = _sources(dataset, labels)
    pipeline.run_perception(sources)
    pipeline.run_identity()
    pipeline.run_phenotype(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)

    target = "animal-01"
    pipeline.run_baseline(
        injections=[
            InjectionSpec(
                feature="step_asymmetry_front",
                magnitude=0.6,
                shape=InjectionShape.STEP,
                starts_at=FIXED_START + timedelta(days=6),
                animal_id=target,
                note="declared synthetic deviation for the P0 stub",
            )
        ]
    )
    events = pipeline.run_events(sources=sources)

    assert pipeline.result.injections, "the run must record what it injected"

    marked = [e for e in events if e.derived_from_injected]
    assert marked, "the injection marker must reach the events"
    assert all(e.animal_id == target for e in marked)

    alerts = [e for e in marked if e.level is EventLevel.ALERT]
    assert alerts, "a deviation of this size must raise an alert"

    # And the marker survives the export.
    exported = json.loads(
        (pipeline.output_root / "export" / f"{alerts[0].idempotency_key}.json").read_text(
            encoding="utf-8"
        )
    )
    assert exported["derived_from_injected"] is True
    assert any("synthetic deviation" in note for note in exported["notes"])

    stored = pipeline.store.read(FEATURES, FeatureRecord)
    assert any(r.injected for r in stored)
    assert isinstance(events[0], HealthEvent)


def test_an_alert_retains_a_masked_clip_and_a_routine_event_does_not(
    tmp_path, pipeline_config, profile, dataset, labels
) -> None:
    pipeline = _pipeline(tmp_path / "clips", pipeline_config, profile, labels)
    sources = _sources(dataset, labels)
    pipeline.run_perception(sources)
    pipeline.run_identity()
    pipeline.run_phenotype(frame_width=FRAME_WIDTH, frame_height=FRAME_HEIGHT)
    pipeline.run_baseline(
        injections=[
            InjectionSpec(
                feature="step_asymmetry_front",
                magnitude=0.6,
                shape=InjectionShape.STEP,
                starts_at=FIXED_START + timedelta(days=6),
                animal_id="animal-02",
            )
        ]
    )
    events = pipeline.run_events(sources=sources)

    alerts = [e for e in events if e.is_alert]
    routine = [e for e in events if not e.is_alert]
    assert alerts and routine

    assert all(e.evidence_clip is None for e in routine)
    retained = [e for e in alerts if e.evidence_clip and e.evidence_clip.retained]
    assert retained, "an alert must retain a clip"
    for event in retained:
        assert event.evidence_clip.masking_applied
        assert Path(event.evidence_clip.path).exists()


# -- 10.5 no media or weights are tracked by version control ----------------


def test_version_control_carries_no_media_or_model_weights() -> None:
    """Covers what is tracked and what a commit would add."""
    import subprocess

    repository = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repository / "tools" / "check_no_media_tracked.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no video, derived media or model weights" in result.stdout


def test_a_clean_clone_contains_no_media_or_model_weights() -> None:
    """What someone else actually receives, rather than what this tree happens to hold."""
    import subprocess

    repository = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [str(repository / "tools" / "check_clean_clone.sh")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "a clean clone contains no media" in result.stdout


def test_the_weights_directory_is_ignored_by_version_control() -> None:
    """Weights are downloaded into the tree; they must never leave it."""
    import subprocess

    repository = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "-q", "weights/model.pt"],
        check=False,
    )
    assert result.returncode == 0, "weights/ must be ignored"


# -- 10.4 the P0 evaluation report ------------------------------------------


def _perception_labels(dataset, labels, pipeline):
    from lhv.evaluation import PerceptionLabels
    from lhv.evaluation.metrics import LabelledBox, LabelledKeypoint
    from lhv.perception import BoundingBox, Tracklet

    boxes, keypoints = [], []
    for relative, entry in labels.items():
        source_id = f"synthetic-lane/{relative}"
        track_id = f"{entry['animal_id']}@{entry['day_key']}"
        for frame in entry["frames"]:
            points = frame["keypoints"]
            if "withers" in points and "base_of_tail" in points:
                cx = (points["withers"][0] + points["base_of_tail"][0]) / 2
                cy = (points["withers"][1] + points["base_of_tail"][1]) / 2
                boxes.append(
                    LabelledBox(
                        frame_index=frame["frame_index"],
                        box=BoundingBox(cx - 30, cy - 48, cx + 30, cy + 48),
                        track_id=track_id,
                        animal_id=entry["animal_id"],
                        source_id=source_id,
                    )
                )
            for name, (x, y) in points.items():
                keypoints.append(
                    LabelledKeypoint(
                        frame_index=frame["frame_index"],
                        name=name,
                        x=x,
                        y=y,
                        visible=True,
                        track_id=track_id,
                        source_id=source_id,
                    )
                )

    identity_truth = {}
    for tracklet in pipeline.store.read("tracklets", Tracklet):
        relative = tracklet.source_id.split("/", 1)[1]
        identity_truth[tracklet.tracklet_id] = labels[relative]["animal_id"]

    return PerceptionLabels(
        boxes=boxes,
        keypoints=keypoints,
        identity_by_tracklet=identity_truth,
        keypoint_normaliser=96.0,
    )


@pytest.fixture
def evaluation(run, dataset, labels):
    from lhv.evaluation import EvaluationHarness, SplitKind

    pipeline, _, _ = run
    harness = EvaluationHarness(pipeline.store, pipeline.series, pipeline.config, pipeline.profile)
    return harness.build(
        split_kind=SplitKind.ANIMAL,
        labels=_perception_labels(dataset, labels, pipeline),
        dataset_name="synthetic-lane",
        dataset_version="1",
    )


def test_the_report_carries_separated_metric_families(evaluation) -> None:
    names = set(evaluation.family_names)
    assert {"detection", "tracking", "pose", "phenotype", "operational"} <= names
    assert {"identity:external_anchor", "identity:visual_fallback"} <= names

    rendered = evaluation.render()
    for family in names:
        assert f"## {family}" in rendered
    assert "No aggregate score is reported across metric families" in rendered


def test_the_report_states_the_stubbed_inference_limitation(evaluation) -> None:
    limitations = " ".join(evaluation.declared_limitations())
    assert "Health inference is stubbed" in limitations
    assert "Nothing in this report is clinical evidence" in limitations


def test_the_report_states_the_site_count(evaluation) -> None:
    assert evaluation.site_count == 1
    limitations = " ".join(evaluation.declared_limitations())
    assert "1 distinct site(s)" in limitations
    assert "Site-disjoint validation was not exercised" in limitations


def test_the_report_is_built_from_an_animal_disjoint_split(evaluation) -> None:
    from lhv.evaluation import SplitKind, assert_no_leakage

    assert evaluation.inputs.split.kind is SplitKind.ANIMAL
    assert evaluation.inputs.split.train_keys and evaluation.inputs.split.test_keys
    assert_no_leakage(evaluation.inputs.split)


def test_perception_metrics_are_reported_against_the_labels(evaluation) -> None:
    detection = evaluation.family("detection")
    tracking = evaluation.family("tracking")
    pose = evaluation.family("pose")

    assert detection.counts["true_positives"] > 0
    assert 0.0 < detection.metrics["precision"] <= 1.0
    assert tracking.counts["identity_switches"] == 0
    assert tracking.metrics["mean_tracklet_purity"] == pytest.approx(1.0)
    assert pose.counts["keypoints_evaluated"] > 0


def test_the_visual_fallback_is_reported_apart_from_the_anchor(evaluation) -> None:
    anchored = evaluation.family("identity:external_anchor")
    fallback = evaluation.family("identity:visual_fallback")
    assert anchored is not None and fallback is not None
    assert anchored.counts["assignments"] > 0
    limitations = " ".join(evaluation.declared_limitations())
    assert "perfect by construction" in limitations


def test_operational_metrics_name_the_policy_and_the_missing_reference(evaluation) -> None:
    operational = evaluation.family("operational")
    assert "alarm_burden_per_1000_animal_days" in operational.metrics
    assert any("p0-default@1" in note for note in operational.notes)
    assert "median_lead_time_days" not in operational.metrics
    assert any("Lead time is unmeasured" in limit for limit in evaluation.declared_limitations())


def test_the_report_is_reproducible_from_its_recorded_inputs(
    evaluation, run, dataset, labels
) -> None:
    from lhv.evaluation import EvaluationHarness, SplitKind

    pipeline, _, _ = run
    again = EvaluationHarness(
        pipeline.store, pipeline.series, pipeline.config, pipeline.profile
    ).build(
        split_kind=SplitKind.ANIMAL,
        labels=_perception_labels(dataset, labels, pipeline),
        dataset_name="synthetic-lane",
        dataset_version="1",
    )
    assert again.fingerprint() == evaluation.fingerprint()
    assert again.render() == evaluation.render()


def test_the_harness_refuses_to_report_from_a_leaking_split(run) -> None:
    from lhv.errors import LeakageError
    from lhv.evaluation import SplitDefinition, SplitKind, assert_no_leakage

    pipeline, _, _ = run
    animals = sorted({o.animal_id for o in pipeline.series.read()})
    leaking = SplitDefinition(
        kind=SplitKind.ANIMAL,
        seed=1,
        test_fraction=0.5,
        train_keys=tuple(animals),
        test_keys=(animals[0],),
    )
    with pytest.raises(LeakageError) as excinfo:
        assert_no_leakage(leaking)
    assert animals[0] in excinfo.value.overlapping
