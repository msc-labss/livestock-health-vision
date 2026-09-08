"""Evaluation harness: splits, leakage, metric families, separation, reproducibility."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from lhv.errors import LeakageError
from lhv.evaluation import (
    EvaluationReport,
    LabelledBox,
    LabelledKeypoint,
    MetricFamily,
    ReportInputs,
    SplitDefinition,
    SplitItem,
    SplitKind,
    assert_no_leakage,
    build_split,
    detection_metrics,
    identity_metrics,
    operational_metrics,
    phenotype_metrics,
    pose_metrics,
    tracking_metrics,
)
from lhv.perception import BoundingBox
from lhv.profiles import load_profile

from .conftest import synthetic_pose_sequence, tracklet_from_poses

BASE = datetime(2024, 3, 1, 6, 30, tzinfo=UTC)


@pytest.fixture
def profile():
    return load_profile("cattle")


def _items(*, animals=6, days=3, sites=1, per=4):
    items = []
    for site in range(sites):
        for animal in range(animals):
            for day in range(days):
                for n in range(per):
                    items.append(
                        SplitItem(
                            item_id=f"s{site}-a{animal}-d{day}-{n}",
                            animal_key=f"a{animal}",
                            day_key=f"d{day}",
                            site_key=f"site-{site}",
                        )
                    )
    return items


# -- 9.1 split construction from provenance ---------------------------------


@pytest.mark.parametrize("kind", [SplitKind.ANIMAL, SplitKind.DAY, SplitKind.SITE])
def test_a_disjoint_split_shares_no_key_across_partitions(kind: SplitKind) -> None:
    items = _items(sites=3)
    split = build_split(items, kind, test_fraction=0.3, seed=11)
    assert set(split.train_keys) & set(split.test_keys) == set()
    assert split.train_keys and split.test_keys
    assert_no_leakage(split, items)


def test_an_animal_disjoint_split_separates_animals() -> None:
    items = _items()
    split = build_split(items, SplitKind.ANIMAL, test_fraction=0.34, seed=3)
    by_id = {i.item_id: i for i in items}
    train = {by_id[i].animal_key for i in split.train_item_ids}
    test = {by_id[i].animal_key for i in split.test_item_ids}
    assert train & test == set()


def test_a_site_disjoint_split_separates_sites() -> None:
    items = _items(sites=4)
    split = build_split(items, SplitKind.SITE, test_fraction=0.25, seed=5)
    by_id = {i.item_id: i for i in items}
    train = {by_id[i].site_key for i in split.train_item_ids}
    test = {by_id[i].site_key for i in split.test_item_ids}
    assert train & test == set()


def test_a_frame_level_split_partitions_individual_items() -> None:
    items = _items(animals=2, days=1, per=10)
    split = build_split(items, SplitKind.FRAME, test_fraction=0.3, seed=1)
    assert set(split.train_item_ids) & set(split.test_item_ids) == set()
    assert len(split.train_item_ids) + len(split.test_item_ids) == len(items)


def test_the_split_is_deterministic_for_a_given_seed() -> None:
    items = _items()
    first = build_split(items, SplitKind.ANIMAL, test_fraction=0.3, seed=42)
    second = build_split(items, SplitKind.ANIMAL, test_fraction=0.3, seed=42)
    other = build_split(items, SplitKind.ANIMAL, test_fraction=0.3, seed=43)
    assert first == second
    assert first.test_keys != other.test_keys or len(set(first.test_keys)) <= 1


def test_a_split_cannot_be_built_on_a_key_the_provenance_lacks() -> None:
    items = [SplitItem(item_id="i1", day_key="d1", site_key="s1")]
    with pytest.raises(LeakageError, match="key absent from provenance"):
        build_split(items, SplitKind.ANIMAL, test_fraction=0.5, seed=1)


# -- 9.2 leakage is a failure, not a warning --------------------------------


def test_an_overlapping_animal_aborts_the_run_naming_the_overlap() -> None:
    leaking = SplitDefinition(
        kind=SplitKind.ANIMAL,
        seed=1,
        test_fraction=0.3,
        train_keys=("a1", "a2", "a3"),
        test_keys=("a3", "a4"),
    )
    with pytest.raises(LeakageError) as excinfo:
        assert_no_leakage(leaking)
    assert excinfo.value.overlapping == ("a3",)
    assert "a3" in str(excinfo.value)


def test_no_metrics_are_produced_from_a_leaking_split(profile) -> None:
    """The harness must refuse before it computes anything."""
    leaking = SplitDefinition(
        kind=SplitKind.ANIMAL,
        seed=1,
        test_fraction=0.3,
        train_keys=("a1",),
        test_keys=("a1",),
    )
    produced = []
    with pytest.raises(LeakageError):
        assert_no_leakage(leaking)
        produced.append(phenotype_metrics([], profile.feature_set.names))
    assert produced == []


def test_leakage_is_detected_from_the_items_even_when_the_key_lists_look_clean() -> None:
    items = [
        SplitItem(item_id="i1", animal_key="a1"),
        SplitItem(item_id="i2", animal_key="a1"),
    ]
    sneaky = SplitDefinition(
        kind=SplitKind.ANIMAL,
        seed=1,
        test_fraction=0.5,
        train_keys=("a1",),
        test_keys=("a2",),
        train_item_ids=("i1",),
        test_item_ids=("i2",),
    )
    with pytest.raises(LeakageError) as excinfo:
        assert_no_leakage(sneaky, items)
    assert excinfo.value.overlapping == ("a1",)


# -- 9.3 perception metric family -------------------------------------------


def _labels_from(poses, *, track_id="gt-1", half=45.0):
    labels = []
    for pose in poses:
        withers = pose.keypoint("withers")
        tail = pose.keypoint("sacrum")
        cx = (withers.x + tail.x) / 2.0
        cy = (withers.y + tail.y) / 2.0
        labels.append(
            LabelledBox(
                frame_index=pose.provenance.frame_index,
                box=BoundingBox(cx - half, cy - half, cx + half, cy + half),
                track_id=track_id,
                animal_id="a1",
                source_id=pose.provenance.source_id,
            )
        )
    return labels


def test_detection_metrics_are_reported(profile) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    tracklet = tracklet_from_poses(poses)
    family = detection_metrics(tracklet.detections, _labels_from(poses), iou_threshold=0.5)

    assert family.name == "detection"
    assert set(family.metrics) >= {"precision", "recall", "f1"}
    assert family.metrics["recall"] == pytest.approx(1.0)
    assert family.counts["true_positives"] == 20


def test_detection_metrics_count_misses_and_false_alarms(profile) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    tracklet = tracklet_from_poses(poses)
    truncated = _labels_from(poses)[:15]
    family = detection_metrics(tracklet.detections, truncated)
    assert family.counts["false_positives"] == 5
    assert family.metrics["precision"] < 1.0


def test_tracking_metrics_are_reported(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    tracklet = tracklet_from_poses(poses)
    family = tracking_metrics([tracklet], _labels_from(poses))

    assert family.name == "tracking"
    assert family.metrics["mean_tracklet_purity"] == pytest.approx(1.0)
    assert family.counts["identity_switches"] == 0
    assert family.metrics["mostly_tracked_fraction"] == pytest.approx(1.0)


def test_tracking_metrics_notice_an_identity_switch(profile) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    tracklet = tracklet_from_poses(poses)
    labels = _labels_from(poses[:10], track_id="gt-1") + _labels_from(poses[10:], track_id="gt-2")
    family = tracking_metrics([tracklet], labels)
    assert family.counts["identity_switches"] == 1
    assert family.metrics["mean_tracklet_purity"] < 1.0


def test_pose_metrics_are_reported(profile) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    labels = [
        LabelledKeypoint(
            frame_index=pose.provenance.frame_index,
            name=keypoint.name,
            x=keypoint.x,
            y=keypoint.y,
            visible=True,
            source_id=pose.provenance.source_id,
        )
        for pose in poses
        for keypoint in pose.keypoints
        if keypoint.observed
    ]
    family = pose_metrics(poses, labels, distance_threshold=0.1, normaliser=80.0)

    assert family.name == "pose"
    assert family.metrics["pck"] == pytest.approx(1.0)
    assert family.counts["keypoints_evaluated"] > 0


def test_pose_metrics_penalise_a_displaced_keypoint(profile) -> None:
    poses = synthetic_pose_sequence(profile, frames=20)
    labels = [
        LabelledKeypoint(
            frame_index=pose.provenance.frame_index,
            name=keypoint.name,
            x=keypoint.x + 60.0,
            y=keypoint.y,
            visible=True,
            source_id=pose.provenance.source_id,
        )
        for pose in poses
        for keypoint in pose.keypoints
        if keypoint.observed
    ]
    family = pose_metrics(poses, labels, distance_threshold=0.1, normaliser=80.0)
    assert family.metrics["pck"] == pytest.approx(0.0)


def test_perception_metrics_run_under_an_animal_disjoint_split(profile, config) -> None:
    """Metrics are computed over the test partition only."""
    poses = synthetic_pose_sequence(profile, frames=20)
    tracklet = tracklet_from_poses(poses)
    labels = _labels_from(poses)

    items = [
        SplitItem(item_id=str(label.frame_index), animal_key="a1", day_key="d1", site_key="site-a")
        for label in labels
    ] + [
        SplitItem(item_id=f"other-{i}", animal_key="a2", day_key="d1", site_key="site-a")
        for i in range(20)
    ]
    split = build_split(items, SplitKind.ANIMAL, test_fraction=0.5, seed=1)
    assert_no_leakage(split, items)

    held_out = {i.item_id for i in items if i.animal_key in split.test_keys}
    predictions = [d for d in tracklet.detections if str(d.frame_index) in held_out]
    truth = [label for label in labels if str(label.frame_index) in held_out]
    family = detection_metrics(predictions, truth)
    assert family.counts["true_positives"] == len(truth)


# -- 9.4 phenotype metric family --------------------------------------------


@dataclasses.dataclass
class _Obs:
    animal_id: str
    features: dict


def test_phenotype_metrics_report_reproducibility_and_stability(profile) -> None:
    observations = []
    for index, animal in enumerate(["a1", "a2", "a3", "a4"]):
        for repeat in range(4):
            observations.append(
                _Obs(
                    animal_id=animal,
                    features={"speed": 0.6 + 0.1 * index + 0.005 * repeat},
                )
            )
    family = phenotype_metrics(observations, ["speed"])

    assert family.name == "phenotype"
    assert "speed.within_animal_cv" in family.metrics
    assert "speed.between_animal_sd" in family.metrics
    assert "speed.icc" in family.metrics
    assert family.counts["animals_with_repeated_passes"] == 4
    # Animals differ far more than a single animal's repeats do.
    assert family.metrics["speed.icc"] > 0.9


def test_a_feature_that_cannot_separate_animals_scores_a_low_icc() -> None:
    import random

    generator = random.Random(0)
    observations = [
        _Obs(animal_id=f"a{a}", features={"noise": generator.gauss(0.0, 1.0)})
        for a in range(6)
        for _ in range(6)
    ]
    family = phenotype_metrics(observations, ["noise"])
    assert family.metrics["noise.icc"] < 0.5


def test_phenotype_metrics_need_no_clinical_label(profile) -> None:
    observations = [
        _Obs(animal_id="a1", features={"speed": 0.7}),
        _Obs(animal_id="a1", features={"speed": 0.72}),
        _Obs(animal_id="a2", features={"speed": 0.9}),
        _Obs(animal_id="a2", features={"speed": 0.91}),
    ]
    family = phenotype_metrics(observations, ["speed"])
    assert family.metrics
    assert any("without any clinical label" in note for note in family.notes)
    assert any("clinical correlation is unevaluated" in note for note in family.notes)


# -- 9.5 the visual fallback is reported standalone -------------------------


def test_the_visual_fallback_is_reported_separately_from_the_anchored_path(config) -> None:
    from lhv.identity import AssignmentMethod, IdentityAssignment

    def assignment(tracklet_id, method, animal_id):
        return IdentityAssignment(
            tracklet_id=tracklet_id,
            method=method,
            assigned_at=BASE,
            site_key="site-a",
            day_key="2024-03-01",
            animal_id=animal_id,
            confidence=0.9,
        )

    assignments = [
        assignment("t1", AssignmentMethod.EXTERNAL_ANCHOR, "a1"),
        assignment("t2", AssignmentMethod.EXTERNAL_ANCHOR, "a2"),
        assignment("t3", AssignmentMethod.VISUAL_FALLBACK, "a3"),
        assignment("t4", AssignmentMethod.VISUAL_FALLBACK, "a9"),
    ]
    truth = {"t1": "a1", "t2": "a2", "t3": "a3", "t4": "a4"}

    anchored = identity_metrics(assignments, truth, method="external_anchor")
    fallback = identity_metrics(assignments, truth, method="visual_fallback")

    assert anchored.name == "identity:external_anchor"
    assert fallback.name == "identity:visual_fallback"
    assert anchored.metrics["accuracy"] == pytest.approx(1.0)
    assert fallback.metrics["accuracy"] == pytest.approx(0.5)
    # The perfect anchor must not be able to hide the fallback's weakness.
    assert anchored.metrics["accuracy"] != fallback.metrics["accuracy"]
    assert any("not pooled" in note for note in fallback.notes)


# -- 9.6 operational metrics are first class --------------------------------


@dataclasses.dataclass
class _Event:
    animal_id: str
    level: str
    event_timestamp: datetime


def test_alarm_burden_is_reported_with_the_policy_that_produced_it() -> None:
    events = [_Event(f"a{i}", "alert", BASE) for i in range(6)] + [
        _Event("a9", "observation", BASE)
    ]
    family = operational_metrics(
        events,
        animal_days=2000,
        policy_identity="p0-default@1",
        reference_event_name="human locomotion score >= 3",
    )
    assert family.name == "operational"
    assert family.metrics["alarm_burden_per_1000_animal_days"] == pytest.approx(3.0)
    assert any("p0-default@1" in note for note in family.notes)


def test_lead_time_names_the_reference_event_it_is_measured_against() -> None:
    events = [_Event("a1", "alert", BASE)]
    family = operational_metrics(
        events,
        animal_days=100,
        policy_identity="p0-default@1",
        reference_event_name="human locomotion score >= 3",
        reference_times={"a1": BASE + timedelta(days=5)},
    )
    assert family.metrics["median_lead_time_days"] == pytest.approx(5.0)
    assert any("human locomotion score >= 3" in note for note in family.notes)


def test_lead_time_is_omitted_rather_than_reported_as_zero() -> None:
    family = operational_metrics(
        [_Event("a1", "alert", BASE)],
        animal_days=100,
        policy_identity="p0-default@1",
        reference_event_name="none available in P0",
    )
    assert "median_lead_time_days" not in family.metrics
    assert any("not reported rather than being reported as zero" in n for n in family.notes)


# -- 9.7, 9.8, 9.9 the report ------------------------------------------------


def _inputs(*, sites=("site-a",)) -> ReportInputs:
    return ReportInputs(
        dataset_name="cattleeyeview",
        dataset_version="2023.1",
        model_identities={"detector": "yolo11m@8.4", "pose": "yolo11m-pose@8.4"},
        split=build_split(_items(), SplitKind.ANIMAL, test_fraction=0.3, seed=7),
        config_digest="abc123",
        feature_set_version="0",
        skeleton="cattleeyeview-topdown-24@1",
        site_keys=tuple(sites),
    )


def _report(**kwargs) -> EvaluationReport:
    report = EvaluationReport(inputs=_inputs(**kwargs))
    report.add(MetricFamily(name="detection", metrics={"precision": 0.8, "recall": 0.7}))
    report.add(MetricFamily(name="phenotype", metrics={"speed.icc": 0.9}))
    report.add(MetricFamily(name="operational", metrics={"alarm_burden_per_1000_animal_days": 3.0}))
    return report


def test_the_report_keeps_families_in_distinct_sections() -> None:
    text = _report().render()
    assert "## detection" in text
    assert "## phenotype" in text
    assert "## operational" in text


def test_the_report_emits_no_aggregate_score_across_families() -> None:
    report = _report()
    rendered = report.render()
    # No reported metric may be a combination across families.
    reported = [line for line in rendered.splitlines() if line.startswith("- ")]
    for line in reported:
        lowered = line.lower()
        assert "overall" not in lowered
        assert "combined" not in lowered
        assert "aggregate" not in lowered
    assert "No aggregate score is reported across metric families" in rendered

    # There is no API that could produce one either.
    assert not hasattr(report, "overall_score")
    assert not hasattr(report, "aggregate")
    keys = [key for family in report.families for key in family.metrics]
    assert len(keys) == len(set(keys)), "metric names must stay namespaced by family"


def test_the_report_records_every_input_it_is_a_function_of() -> None:
    text = _report().render()
    assert "cattleeyeview@2023.1" in text
    assert "abc123" in text
    assert "yolo11m@8.4" in text
    assert "animal-disjoint" in text
    assert "cattleeyeview-topdown-24@1" in text


def test_the_same_inputs_produce_the_same_report() -> None:
    first = _report()
    second = _report()
    assert first.fingerprint() == second.fingerprint()
    assert first.content() == second.content()
    assert first.render() == second.render()


def test_a_different_input_produces_a_different_report() -> None:
    changed = EvaluationReport(inputs=dataclasses.replace(_inputs(), config_digest="different"))
    changed.add(MetricFamily(name="detection", metrics={"precision": 0.8, "recall": 0.7}))
    changed.add(MetricFamily(name="phenotype", metrics={"speed.icc": 0.9}))
    changed.add(
        MetricFamily(name="operational", metrics={"alarm_burden_per_1000_animal_days": 3.0})
    )
    assert changed.fingerprint() != _report().fingerprint()


def test_the_report_is_regenerated_identically_from_its_recorded_inputs() -> None:
    original = _report()
    recorded = original.content()["inputs"]

    regenerated = EvaluationReport(
        inputs=ReportInputs(
            dataset_name=recorded["dataset_name"],
            dataset_version=recorded["dataset_version"],
            model_identities=recorded["model_identities"],
            split=SplitDefinition.from_dict(recorded["split"]),
            config_digest=recorded["config_digest"],
            feature_set_version=recorded["feature_set_version"],
            skeleton=recorded["skeleton"],
            site_keys=tuple(recorded["site_keys"]),
        )
    )
    for family in original.families:
        regenerated.add(family)
    assert regenerated.fingerprint() == original.fingerprint()
    assert regenerated.render() == original.render()


def test_the_generation_time_does_not_change_the_report() -> None:
    first = _report()
    first.generated_at = datetime(2024, 1, 1, tzinfo=UTC)
    second = _report()
    second.generated_at = datetime(2025, 6, 6, tzinfo=UTC)
    assert first.fingerprint() == second.fingerprint()


def test_every_report_states_that_health_inference_is_stubbed() -> None:
    limitations = " ".join(_report().declared_limitations())
    assert "Health inference is stubbed" in limitations
    assert "Nothing in this report is clinical evidence" in limitations
    assert "Health inference is stubbed" in _report().render()


def test_a_single_site_evaluation_states_the_site_count_and_the_gap() -> None:
    report = _report(sites=("site-a",))
    assert report.site_count == 1
    limitations = " ".join(report.declared_limitations())
    assert "1 distinct site(s)" in limitations
    assert "Site-disjoint validation was not exercised" in limitations


def test_a_multi_site_evaluation_does_not_claim_the_gap() -> None:
    report = _report(sites=("site-a", "site-b"))
    assert report.site_count == 2
    limitations = " ".join(report.declared_limitations())
    assert "2 distinct site(s)" in limitations
    assert "Site-disjoint validation was not exercised" not in limitations


def test_a_report_containing_injected_data_says_so_and_lists_it() -> None:
    report = _report()
    report.injections = ["head_bob: +0.40 as a step on a1 from 2024-03-05T00:00:00"]
    rendered = report.render()
    assert "Injected data was present" in rendered
    assert "head_bob: +0.40" in rendered


# -- metrics are keyed by source, not by frame index alone ------------------


def test_labels_from_one_source_do_not_score_another_source(profile) -> None:
    """Frame indices restart at zero in every source.

    Keying on the frame index alone silently scores each sequence against every
    other sequence's labels, which inflates or destroys every perception number
    depending on how the fixtures happen to line up.
    """
    first = synthetic_pose_sequence(profile, frames=20, source_id="src-a")
    second = synthetic_pose_sequence(profile, frames=20, source_id="src-b", speed=-6.0)

    labels = [
        dataclasses.replace(label, source_id="src-a")
        for label in _labels_from(first, track_id="gt-a")
    ] + [
        dataclasses.replace(label, source_id="src-b")
        for label in _labels_from(second, track_id="gt-b")
    ]

    tracklets = [
        tracklet_from_poses(first, tracklet_id="src-a:t0"),
        tracklet_from_poses(second, tracklet_id="src-b:t0"),
    ]
    detections = [d for t in tracklets for d in t.detections]

    family = detection_metrics(detections, labels)
    assert family.counts["false_positives"] == 0
    assert family.counts["false_negatives"] == 0
    assert family.metrics["precision"] == pytest.approx(1.0)

    tracking = tracking_metrics(tracklets, labels)
    assert tracking.counts["identity_switches"] == 0
    assert tracking.metrics["mean_tracklet_purity"] == pytest.approx(1.0)
    assert tracking.metrics["mostly_tracked_fraction"] == pytest.approx(1.0)


def test_pose_labels_are_keyed_by_source_too(profile) -> None:
    first = synthetic_pose_sequence(profile, frames=10, source_id="src-a")
    second = synthetic_pose_sequence(profile, frames=10, source_id="src-b", speed=-6.0)

    labels = [
        LabelledKeypoint(
            frame_index=pose.provenance.frame_index,
            name=keypoint.name,
            x=keypoint.x,
            y=keypoint.y,
            visible=True,
            source_id=pose.provenance.source_id,
        )
        for pose in first + second
        for keypoint in pose.keypoints
        if keypoint.observed
    ]
    family = pose_metrics(first + second, labels, distance_threshold=0.05, normaliser=80.0)
    assert family.metrics["pck"] == pytest.approx(1.0)


# -- a report states how much of the feature set it could not compute ---------


def test_the_report_states_which_declared_features_were_unavailable() -> None:
    """Phenotype metrics over three features are a smaller claim than over nine.

    "Declared limitations appear in the report" already required this; six of
    nine features being uncomputable is a limitation that applies, and the
    report did not say so.
    """
    from lhv.evaluation.report import EvaluationReport, ReportInputs
    from lhv.evaluation.splits import SplitDefinition
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    unavailable = {
        f.name: " ".join(f.unavailable_reason.split()) for f in profile.feature_set.unavailable
    }
    assert unavailable, "this test needs an unavailable feature to mean anything"

    report = EvaluationReport(
        inputs=ReportInputs(
            dataset_name="d",
            dataset_version="1",
            model_identities={},
            split=SplitDefinition(
                kind="animal", train_keys=("a",), test_keys=("b",), seed=1, test_fraction=0.5
            ),
            config_digest="x",
            feature_set_version=profile.feature_set.version,
            skeleton="s",
            site_keys=("one",),
            unavailable_features=unavailable,
            feature_names=profile.feature_set.names,
        )
    )
    stated = "\n".join(report.declared_limitations())

    assert f"{len(unavailable)} of {len(profile.feature_set)} declared features" in stated
    for name, reason in unavailable.items():
        assert name in stated
        assert reason[:30] in stated, f"{name} is listed without its reason"


def test_unavailable_features_change_the_report_fingerprint() -> None:
    """Two runs differing only in what they could compute are not the same evaluation."""
    from lhv.evaluation.report import ReportInputs
    from lhv.evaluation.splits import SplitDefinition

    def inputs(**overrides):
        base = dict(
            dataset_name="d",
            dataset_version="1",
            model_identities={},
            split=SplitDefinition(
                kind="animal", train_keys=("a",), test_keys=("b",), seed=1, test_fraction=0.5
            ),
            config_digest="x",
            feature_set_version="1",
            skeleton="s",
            site_keys=("one",),
        )
        return ReportInputs(**{**base, **overrides})

    everything = inputs()
    missing_one = inputs(unavailable_features={"back_posture": "no mid-dorsal keypoint"})
    assert everything.fingerprint() != missing_one.fingerprint()
