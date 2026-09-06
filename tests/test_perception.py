"""Perception: record schemas, detection, model identity, tracking, pose."""

from __future__ import annotations

import numpy as np
import pytest

from lhv.errors import MissingFieldError, ViewMismatchError
from lhv.ingest import Frame, Ingestor
from lhv.perception import (
    AnnotationPoseBackend,
    BoundingBox,
    Detection,
    Detector,
    FrameDetections,
    IntensityBlobDetector,
    Keypoint,
    NativeKeypoint,
    Pose,
    PoseEstimator,
    TerminationReason,
    Tracker,
    Tracklet,
    Visibility,
    map_to_skeleton,
)
from lhv.profiles import load_profile


@pytest.fixture
def profile():
    return load_profile("cattle")


def _provenance(index: int = 0, **overrides):
    from datetime import UTC, datetime

    from lhv.ingest import FrameProvenance

    base = dict(
        source_id="s",
        camera_id="c",
        frame_index=index,
        site_key="site-a",
        animal_set_key="herd-a",
        day_key="2024-03-01",
        capture_timestamp=datetime(2024, 3, 1, 6, 30, index, tzinfo=UTC),
    )
    base.update(overrides)
    return FrameProvenance(**base)


def _detection(index: int, box: BoundingBox, *, confidence: float = 0.9, suffix: str = "0"):
    return Detection(
        detection_id=f"s#{index}:{suffix}",
        provenance=_provenance(index),
        box=box,
        label="animal",
        confidence=confidence,
        model_identity="stub@1",
    )


def _frame_record(index: int, detections, width: int = 320, height: int = 240):
    return FrameDetections(
        provenance=_provenance(index),
        model_identity="stub@1",
        detections=tuple(detections),
        frame_width=width,
        frame_height=height,
    )


# -- 4.1 stage-boundary schemas ---------------------------------------------


@pytest.mark.parametrize(
    ("klass", "expected"),
    [
        (Detection, "detection"),
        (FrameDetections, "frame_detections"),
        (Pose, "pose"),
        (Tracklet, "tracklet"),
    ],
)
def test_every_perception_schema_carries_its_own_version(klass, expected) -> None:
    assert klass.SCHEMA_NAME == expected
    assert klass.SCHEMA_VERSION


def test_detection_records_its_schema_version_on_the_record() -> None:
    detection = _detection(0, BoundingBox(0, 0, 10, 10))
    encoded = detection.to_dict()
    assert encoded["schema_name"] == "detection"
    assert encoded["schema_version"] == Detection.SCHEMA_VERSION


@pytest.mark.parametrize(
    ("klass", "kwargs", "missing"),
    [
        (
            Detection,
            dict(
                detection_id="d",
                provenance=_provenance(),
                label="animal",
                confidence=0.9,
                model_identity="m@1",
            ),
            "box",
        ),
        (FrameDetections, dict(provenance=_provenance()), "model_identity"),
        (
            Pose,
            dict(
                pose_id="p",
                provenance=_provenance(),
                detection_id="d",
                skeleton_id="sk",
                skeleton_version="1",
                model_identity="m@1",
            ),
            "keypoints",
        ),
        (
            Tracklet,
            dict(
                tracklet_id="t",
                source_id="s",
                camera_id="c",
                site_key="site-a",
                animal_set_key="herd-a",
                day_key="2024-03-01",
                first_frame_index=0,
                last_frame_index=1,
                termination_reason=TerminationReason.EXIT,
                model_identity="m@1",
            ),
            "detections",
        ),
    ],
)
def test_schema_rejects_a_record_missing_a_required_field(klass, kwargs, missing) -> None:
    with pytest.raises(MissingFieldError) as excinfo:
        klass(**kwargs)
    assert excinfo.value.field_name == missing


def test_records_round_trip_through_their_own_encoding() -> None:
    detection = _detection(3, BoundingBox(1, 2, 30, 40))
    restored = Detection.from_dict(detection.to_dict())
    assert restored == detection


# -- 4.2 detection ----------------------------------------------------------


def test_detection_carries_frame_provenance(source, config) -> None:
    detector = Detector(IntensityBlobDetector(), config)
    records = detector.detect_all(Ingestor(source, config).iter_frames())
    detected = [r for r in records if not r.is_empty]
    assert detected, "the synthetic lane should produce detections"
    for record in detected:
        for detection in record.detections:
            assert detection.provenance.source_id == source.source_id
            assert detection.provenance.site_key == source.site_key
            assert detection.provenance.day_key
            assert detection.provenance.camera_id == source.camera_id


def test_frame_with_no_animal_yields_an_empty_result_not_an_absent_one(config) -> None:
    detector = Detector(IntensityBlobDetector(), config)
    blank = Frame(provenance=_provenance(7), image=np.full((240, 320, 3), 20, dtype=np.uint8))
    record = detector.detect_frame(blank)
    assert record.is_empty
    assert len(record) == 0
    assert record.provenance.frame_index == 7
    assert detector.frames_processed == 1
    assert detector.frames_without_detection == 1


def test_every_frame_appears_in_the_stream_including_empty_ones(tmp_path, config) -> None:
    from lhv.ingest import register_source

    from .conftest import FIXED_START, write_video

    path = write_video(tmp_path / "m" / "lane.avi", frames=16, absent_from=10)
    source = register_source(
        source_id="synthetic/vanishing",
        camera_id="cam-1",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(path),
        start_timestamp=FIXED_START,
    )
    detector = Detector(IntensityBlobDetector(), config)
    records = detector.detect_all(Ingestor(source, config).iter_frames())
    assert len(records) == 16
    assert [r.provenance.frame_index for r in records] == list(range(16))
    assert any(r.is_empty for r in records)


# -- 4.3 model identity -----------------------------------------------------


def test_every_output_records_the_model_that_produced_it(source, config) -> None:
    detector = Detector(IntensityBlobDetector(version="1"), config)
    records = detector.detect_all(Ingestor(source, config).iter_frames())
    for record in records:
        assert record.model_identity == "intensity-blob@1"
        for detection in record.detections:
            assert detection.model_identity == "intensity-blob@1"


def test_swapping_weights_makes_outputs_distinguishable_by_identity_alone(source, config) -> None:
    frames = list(Ingestor(source, config).iter_frames())

    first = Detector(IntensityBlobDetector(version="1"), config).detect_all(frames)
    second = Detector(IntensityBlobDetector(version="2"), config).detect_all(frames)

    identities = {d.model_identity for r in first for d in r.detections}
    other = {d.model_identity for r in second for d in r.detections}
    assert identities == {"intensity-blob@1"}
    assert other == {"intensity-blob@2"}
    assert identities.isdisjoint(other)


def test_tracklets_and_poses_record_a_model_identity_too(config, profile) -> None:
    detections = [_detection(i, BoundingBox(100 + i, 100, 140 + i, 150)) for i in range(6)]
    tracklets = Tracker(config, source_id="s").track(
        [_frame_record(i, [d]) for i, d in enumerate(detections)]
    )
    assert tracklets and all(t.model_identity == "stub@1" for t in tracklets)

    estimator = PoseEstimator(
        AnnotationPoseBackend({}), profile, config, source_id="s", source_view=profile.skeleton.view
    )
    pose = estimator.estimate(np.zeros((240, 320, 3), np.uint8), detections[0])
    assert pose.model_identity == "annotation@1"


# -- placeholder weights are identifiable on their output ---------------------


def test_the_profile_declares_which_weights_are_placeholders(profile) -> None:
    placeholders = profile.placeholder_weights()
    assert placeholders, "the profile declares no placeholder weights"
    for reference in placeholders:
        assert reference.placeholder is True
        assert reference.placeholder_reason.strip(), (
            f"{reference.role} is a placeholder with no stated reason"
        )


def test_placeholder_status_rides_the_model_identity(profile) -> None:
    from lhv.config import ModelIdentity

    reference = profile.weight("pose")
    identity = ModelIdentity(
        name=reference.name,
        version=reference.version,
        task="pose",
        placeholder=reference.placeholder,
        placeholder_reason=reference.placeholder_reason,
    )
    assert identity.placeholder is reference.placeholder
    assert identity.placeholder_reason == reference.placeholder_reason


def test_a_report_names_placeholder_weights_and_says_what_they_measure() -> None:
    from lhv.perception.report import PerceptionReport

    marked = PerceptionReport(
        source_id="s",
        placeholder_weights={"pose": "not trained on cattle"},
    )
    described = marked.describe()
    assert "PLACEHOLDER" in described
    assert "pose" in described
    assert "not trained on cattle" in described
    assert "not an achievable result" in described


def test_replacing_placeholder_weights_removes_the_mark() -> None:
    from lhv.perception.report import PerceptionReport

    replaced = PerceptionReport(source_id="s", placeholder_weights={})
    assert "PLACEHOLDER" not in replaced.describe()
    marked = PerceptionReport(source_id="s", placeholder_weights={"pose": "analogy map"})
    # The two runs are distinguishable by that field alone.
    assert marked.placeholder_weights != replaced.placeholder_weights
    assert marked.describe() != replaced.describe()


# -- the skeleton's view must match the source's -----------------------------


def test_a_source_recorded_under_another_view_aborts_before_any_keypoint(config, profile) -> None:
    foreign = "lateral" if profile.skeleton.view != "lateral" else "top-down"
    with pytest.raises(ViewMismatchError) as raised:
        PoseEstimator(
            AnnotationPoseBackend({}), profile, config, source_id="ramp/01", source_view=foreign
        )
    message = str(raised.value)
    assert foreign in message and profile.skeleton.view in message
    assert raised.value.source_id == "ramp/01"
    assert raised.value.skeleton_id == profile.skeleton.identifier


def test_a_source_declaring_no_view_is_refused_rather_than_assumed(config, profile) -> None:
    with pytest.raises(ViewMismatchError) as raised:
        PoseEstimator(
            AnnotationPoseBackend({}), profile, config, source_id="ramp/02", source_view=""
        )
    assert "ramp/02" in str(raised.value)
    assert "declares no view" in str(raised.value)
    assert raised.value.source_view == ""


def test_a_matching_view_proceeds_and_is_recorded_on_the_pose(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers"])}
    )
    estimator = PoseEstimator(
        backend, profile, config, source_id="ramp/03", source_view=profile.skeleton.view
    )
    pose = estimator.estimate(np.zeros((240, 320, 3), np.uint8), detection)
    assert pose.view == profile.skeleton.view


# -- 4.4 tracklet formation and termination reasons -------------------------


def test_consecutive_detections_form_one_tracklet(config) -> None:
    records = [
        _frame_record(i, [_detection(i, BoundingBox(100, 100 + i * 4, 140, 150 + i * 4))])
        for i in range(10)
    ]
    tracklets = Tracker(config, source_id="s").track(records)
    assert len(tracklets) == 1
    assert tracklets[0].length == 10
    assert tracklets[0].frame_indices == tuple(range(10))


def test_exit_is_recorded_when_the_animal_leaves_the_field_of_view(config) -> None:
    """Walk to the frame edge, then stop being detected."""
    records = []
    for i in range(8):
        y = 100 + i * 20
        records.append(_frame_record(i, [_detection(i, BoundingBox(100, y, 140, y + 50))]))
    records.extend(_frame_record(i, []) for i in range(8, 40))

    tracklets = Tracker(config, source_id="s").track(records)
    assert len(tracklets) == 1
    assert tracklets[0].termination_reason is TerminationReason.EXIT


def test_occlusion_loss_is_distinguished_from_detection_failure(config) -> None:
    """A second animal walks over the first, which then stops being detected.

    From a motion tracker's point of view that is exactly the difference between
    an occlusion and a detector miss: something else is standing where the lost
    track was predicted to be.
    """
    records = []
    # The occluded animal, drifting slowly; the passer-by, moving down the lane.
    for i in range(6):
        occluded = _detection(i, BoundingBox(100, 100 + i, 140, 150 + i), suffix="a")
        passer = _detection(i, BoundingBox(100, -140 + 40 * i, 140, -90 + 40 * i), suffix="b")
        records.append(_frame_record(i, [passer, occluded]))
    # From here only the passer-by is detected, and at frame 6 it is exactly
    # where the occluded track was predicted to be.
    for i in range(6, 40):
        records.append(
            _frame_record(
                i, [_detection(i, BoundingBox(100, -140 + 40 * i, 140, -90 + 40 * i), suffix="b")]
            )
        )

    tracklets = Tracker(config, source_id="s").track(records)
    by_reason = {t.termination_reason for t in tracklets}
    assert TerminationReason.OCCLUSION_LOSS in by_reason
    assert TerminationReason.DETECTION_FAILURE not in by_reason


def test_detection_failure_is_recorded_when_nothing_explains_the_loss(config) -> None:
    records = [_frame_record(i, [_detection(i, BoundingBox(100, 100, 140, 150))]) for i in range(6)]
    records.extend(_frame_record(i, []) for i in range(6, 40))

    tracklets = Tracker(config, source_id="s").track(records)
    assert len(tracklets) == 1
    assert tracklets[0].termination_reason is TerminationReason.DETECTION_FAILURE


def test_the_three_loss_reasons_are_distinct_values() -> None:
    values = {
        TerminationReason.EXIT.value,
        TerminationReason.OCCLUSION_LOSS.value,
        TerminationReason.DETECTION_FAILURE.value,
    }
    assert len(values) == 3


def test_source_end_is_not_reported_as_a_loss(config) -> None:
    records = [_frame_record(i, [_detection(i, BoundingBox(100, 100, 140, 150))]) for i in range(5)]
    tracklets = Tracker(config, source_id="s").track(records)
    assert tracklets[0].termination_reason is TerminationReason.SOURCE_END


def test_tracker_reports_termination_reasons_for_the_run(config) -> None:
    records = [_frame_record(i, [_detection(i, BoundingBox(100, 100, 140, 150))]) for i in range(5)]
    tracker = Tracker(config, source_id="s")
    tracker.track(records)
    assert tracker.report.tracklets_formed == 1
    assert tracker.report.terminations == {"source_end": 1}


# -- 4.5 tracklet identifier uniqueness -------------------------------------


def test_tracklet_identifiers_are_unique_within_a_source(config) -> None:
    records = []
    for i in range(30):
        detections = [_detection(i, BoundingBox(20, 20 + i * 2, 60, 70 + i * 2), suffix="a")]
        if i % 7 == 0:
            detections.append(
                _detection(i, BoundingBox(200, 20 + i * 3, 250, 80 + i * 3), suffix="b")
            )
        records.append(_frame_record(i, detections))

    tracklets = Tracker(config, source_id="s").track(records)
    identifiers = [t.tracklet_id for t in tracklets]
    assert len(identifiers) == len(set(identifiers))
    assert len(tracklets) > 1, "the fixture must actually produce more than one tracklet"


def test_tracklet_identifiers_are_unique_over_a_full_synthetic_pass(source, config) -> None:
    frames = list(Ingestor(source, config).iter_frames())
    records = Detector(IntensityBlobDetector(), config).detect_all(frames)
    tracklets = Tracker(config, source_id=source.source_id).track(records)
    identifiers = [t.tracklet_id for t in tracklets]
    assert identifiers
    assert len(identifiers) == len(set(identifiers))
    assert all(t.tracklet_id.startswith(source.source_id) for t in tracklets)


# -- 4.6 pose against the profile skeleton ----------------------------------


def _annotation_for(detection: Detection, names, *, confidence: float = 0.9):
    cx, cy = detection.box.centre
    return [
        NativeKeypoint(name=name, x=cx + i, y=cy + i, confidence=confidence)
        for i, name in enumerate(names)
    ]


def test_pose_output_records_the_skeleton_identifier_and_version(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers", "sacrum"])}
    )
    pose = PoseEstimator(
        backend, profile, config, source_id="s", source_view=profile.skeleton.view
    ).estimate(np.zeros((240, 320, 3), np.uint8), detection)
    assert pose.skeleton_id == profile.skeleton.identifier
    assert pose.skeleton_version == profile.skeleton.version
    assert len(pose.keypoints) == len(profile.skeleton)


def test_pose_emits_every_profile_keypoint_by_name(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers"])}
    )
    pose = PoseEstimator(
        backend, profile, config, source_id="s", source_view=profile.skeleton.view
    ).estimate(np.zeros((240, 320, 3), np.uint8), detection)
    assert [k.name for k in pose.keypoints] == list(profile.skeleton.names)


def test_a_backend_with_a_foreign_convention_is_mapped_onto_the_skeleton(profile) -> None:
    native = [
        NativeKeypoint(name="left_wrist", x=10, y=20, confidence=0.9),
        NativeKeypoint(name="nose", x=5, y=5, confidence=0.8),
        NativeKeypoint(name="left_earlobe", x=1, y=1, confidence=0.9),
    ]
    keypoints = map_to_skeleton(
        native,
        skeleton=profile.skeleton,
        keypoint_map={"left_wrist": "left_front_hoof", "nose": "nose"},
        visibility_threshold=0.3,
    )
    by_name = {k.name: k for k in keypoints}
    assert by_name["left_front_hoof"].visibility is Visibility.VISIBLE
    assert by_name["nose"].visibility is Visibility.VISIBLE
    # A native keypoint with no counterpart is dropped, not forced onto a name.
    assert by_name["withers"].visibility is Visibility.NOT_VISIBLE


# -- 4.7 keypoint visibility ------------------------------------------------


def test_an_unobservable_keypoint_is_not_visible_and_carries_no_coordinate(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers", "sacrum"])}
    )
    pose = PoseEstimator(
        backend, profile, config, source_id="s", source_view=profile.skeleton.view
    ).estimate(np.zeros((240, 320, 3), np.uint8), detection)
    unobserved = [k for k in pose.keypoints if k.name not in {"withers", "sacrum"}]
    assert unobserved
    for keypoint in unobserved:
        assert keypoint.visibility is Visibility.NOT_VISIBLE
        assert keypoint.x is None and keypoint.y is None
        assert not keypoint.observed


def test_a_low_confidence_keypoint_is_not_presented_as_an_observation(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers"], confidence=0.05)}
    )
    pose = PoseEstimator(
        backend, profile, config, source_id="s", source_view=profile.skeleton.view
    ).estimate(np.zeros((240, 320, 3), np.uint8), detection)
    withers = pose.keypoint("withers")
    assert withers.visibility is Visibility.NOT_VISIBLE
    assert withers.x is None
    assert withers.confidence == pytest.approx(0.05)


def test_a_keypoint_record_refuses_a_coordinate_it_did_not_observe() -> None:
    with pytest.raises(ValueError, match="not visible but carries a coordinate"):
        Keypoint(name="withers", x=1.0, y=2.0, confidence=0.0, visibility=Visibility.NOT_VISIBLE)


# -- 4.8 explicit degradation ------------------------------------------------


def test_low_confidence_detections_remain_in_the_stream_and_are_counted(config) -> None:
    import dataclasses

    tuned = dataclasses.replace(
        config,
        perception=dataclasses.replace(
            config.perception,
            detection_threshold=0.01,
            detection_low_confidence_threshold=0.99,
        ),
    )
    detector = Detector(IntensityBlobDetector(), tuned)
    image = np.full((240, 320, 3), 20, dtype=np.uint8)
    image[100:130, 100:140] = 255
    record = detector.detect_frame(Frame(provenance=_provenance(0), image=image))

    assert not record.is_empty, "a low-confidence detection stays in the stream"
    assert all(d.low_confidence for d in record.detections)
    assert detector.low_confidence_detections == len(record.detections)


def test_low_confidence_poses_are_marked_and_counted(config, profile) -> None:
    detection = _detection(0, BoundingBox(100, 100, 160, 200))
    backend = AnnotationPoseBackend(
        {detection.detection_id: _annotation_for(detection, ["withers"], confidence=0.35)}
    )
    estimator = PoseEstimator(
        backend, profile, config, source_id="s", source_view=profile.skeleton.view
    )
    pose = estimator.estimate(np.zeros((240, 320, 3), np.uint8), detection)
    assert pose.low_confidence
    assert estimator.low_confidence_poses == 1
    assert estimator.keypoints_not_visible == len(profile.skeleton) - 1


def test_run_summary_reports_the_counts(config, profile) -> None:
    from lhv.perception import PerceptionReport

    report = PerceptionReport(
        source_id="s",
        detector_identity="intensity-blob@1",
        pose_identity="annotation@1",
        skeleton=f"{profile.skeleton.identifier}@{profile.skeleton.version}",
        frames_processed=10,
        frames_without_detection=2,
        detections_emitted=8,
        detections_low_confidence=3,
        poses_emitted=8,
        poses_low_confidence=1,
        keypoints_emitted=192,
        keypoints_not_visible=180,
    )
    text = report.describe()
    assert "3 marked low confidence" in text
    assert "180 not visible" in text
    assert "2 with no detection" in text


# -- a detector that serves labels rather than predicting -------------------


def test_the_annotation_detector_serves_the_boxes_for_its_own_frame(config) -> None:
    from lhv.perception import AnnotationDetector

    boxes = {
        ("s", 0): [(BoundingBox(10, 10, 60, 70), "animal")],
        ("s", 1): [
            (BoundingBox(12, 14, 62, 74), "animal"),
            (BoundingBox(200, 20, 250, 80), "animal"),
        ],
    }
    detector = Detector(AnnotationDetector(boxes), config)

    first = detector.detect_frame(
        Frame(provenance=_provenance(0), image=np.zeros((240, 320, 3), np.uint8))
    )
    second = detector.detect_frame(
        Frame(provenance=_provenance(1), image=np.zeros((240, 320, 3), np.uint8))
    )
    third = detector.detect_frame(
        Frame(provenance=_provenance(2), image=np.zeros((240, 320, 3), np.uint8))
    )

    assert len(first) == 1
    assert len(second) == 2
    assert third.is_empty, "a frame with no label yields an empty result, not an absent one"
    assert first.model_identity == "dataset-box-label@1"


def test_the_annotation_detector_says_so_in_its_identity(config) -> None:
    """Labels must never be reportable as a detector's output."""
    from lhv.perception import AnnotationDetector

    identity = AnnotationDetector({}, source="dataset-box-label", version="2").model_identity
    assert identity == "dataset-box-label@2"
    assert "label" in identity


def test_the_annotation_detector_refuses_a_frame_it_cannot_identify() -> None:
    from lhv.perception import AnnotationDetector

    with pytest.raises(ValueError, match="which frame it is looking at"):
        AnnotationDetector({}).detect(np.zeros((10, 10, 3), np.uint8), None)


def test_labelled_boxes_still_go_through_the_tracker(config) -> None:
    """Boxes are supplied; grouping them into tracklets is not."""
    from lhv.perception import AnnotationDetector

    boxes = {
        ("s", i): [(BoundingBox(100, 100 + i * 4, 140, 150 + i * 4), "animal")] for i in range(10)
    }
    detector = Detector(AnnotationDetector(boxes), config)
    records = [
        detector.detect_frame(
            Frame(provenance=_provenance(i), image=np.zeros((240, 320, 3), np.uint8))
        )
        for i in range(10)
    ]
    tracklets = Tracker(config, source_id="s").track(records)
    assert len(tracklets) == 1
    assert tracklets[0].length == 10
