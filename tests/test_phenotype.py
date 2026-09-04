"""Gait phenotype: pass segmentation, the versioned feature set, quality, validity."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from lhv.errors import UndeclaredFeatureError
from lhv.phenotype import (
    FeatureExtractor,
    FeatureRecord,
    PassCompleteness,
    QualityFlag,
    ValidityReason,
    segment_passes,
)
from lhv.profiles import load_profile

from .conftest import synthetic_pose_sequence, tracklet_from_poses

FRAME = dict(frame_width=320, frame_height=240)


@pytest.fixture
def profile():
    return load_profile("cattle")


@pytest.fixture
def poses(profile):
    return synthetic_pose_sequence(profile, frames=40)


@pytest.fixture
def tracklet(poses):
    return tracklet_from_poses(poses)


# -- 6.1 pass segmentation --------------------------------------------------


def test_a_complete_pass_spans_exactly_the_two_boundary_crossings(tracklet, config) -> None:
    complete = [
        p
        for p in segment_passes(tracklet, config, **FRAME)
        if p.completeness is PassCompleteness.COMPLETE
    ]
    assert len(complete) == 1
    lane_pass = complete[0]
    assert lane_pass.entry_frame_index == lane_pass.first_frame_index
    assert lane_pass.exit_frame_index == lane_pass.last_frame_index
    assert lane_pass.missing_boundary == ""


def test_boundaries_are_configuration_not_a_built_in_geometry(tracklet, config) -> None:
    narrow = dataclasses.replace(
        config,
        phenotype=dataclasses.replace(config.phenotype, entry_boundary=0.4, exit_boundary=0.6),
    )
    wide_pass = next(
        p
        for p in segment_passes(tracklet, config, **FRAME)
        if p.completeness is PassCompleteness.COMPLETE
    )
    narrow_pass = next(
        p
        for p in segment_passes(tracklet, narrow, **FRAME)
        if p.completeness is PassCompleteness.COMPLETE
    )
    assert narrow_pass.frame_count < wide_pass.frame_count


def test_a_lane_travelled_in_the_other_direction_still_segments(profile, config) -> None:
    reversed_config = dataclasses.replace(
        config,
        phenotype=dataclasses.replace(config.phenotype, entry_boundary=0.8, exit_boundary=0.2),
    )
    poses = synthetic_pose_sequence(profile, frames=40, speed=-6.0)
    # Start high in the frame so travelling in -y crosses 0.8 then 0.2.
    shifted = []
    for pose in poses:
        moved = [dataclasses.replace(k, y=k.y + 220.0) if k.observed else k for k in pose.keypoints]
        shifted.append(dataclasses.replace(pose, keypoints=tuple(moved)))
    tracklet = tracklet_from_poses(shifted)
    passes = segment_passes(tracklet, reversed_config, **FRAME)
    assert any(p.completeness is PassCompleteness.COMPLETE for p in passes)
    assert all(p.direction == "reverse" for p in passes)


# -- 6.2 partial passes are labelled, not extrapolated ----------------------


def test_a_pass_that_begins_after_the_entry_boundary_is_partial(profile, config) -> None:
    """The animal is already inside the lane when tracking starts."""
    poses = synthetic_pose_sequence(profile, frames=14, speed=6.0)
    shifted = []
    for pose in poses:
        moved = [dataclasses.replace(k, y=k.y + 90.0) if k.observed else k for k in pose.keypoints]
        shifted.append(dataclasses.replace(pose, keypoints=tuple(moved)))
    passes = segment_passes(tracklet_from_poses(shifted), config, **FRAME)

    assert passes[0].completeness is PassCompleteness.PARTIAL
    assert passes[0].entry_frame_index is None
    assert "entry" in passes[0].missing_boundary


def test_a_pass_that_ends_before_the_exit_boundary_is_partial(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=12, speed=6.0)
    passes = segment_passes(tracklet_from_poses(poses), config, **FRAME)
    assert passes[0].completeness is PassCompleteness.PARTIAL
    assert passes[0].exit_frame_index is None
    assert "exit" in passes[0].missing_boundary


def test_a_partial_pass_does_not_invent_the_missing_crossing(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=12, speed=6.0)
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    assert lane_pass.exit_frame_index is None
    assert lane_pass.last_frame_index == poses[-1].provenance.frame_index


# -- 6.3 versioned named feature set ----------------------------------------


def test_the_feature_record_names_its_feature_set_version(tracklet, poses, profile, config) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert record.feature_set_version == profile.feature_set.version
    assert record.to_dict()["feature_set_version"] == profile.feature_set.version


def test_every_declared_feature_is_emitted_with_its_unit(tracklet, poses, profile, config) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert [f.name for f in record.features] == list(profile.feature_set.names)
    for feature in record.features:
        assert feature.unit == profile.feature_set.get(feature.name).unit


def test_features_are_computed_and_finite_for_a_clean_pass(
    tracklet, poses, profile, config
) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert record.valid
    for feature in record.features:
        assert np.isfinite(feature.value), f"{feature.name} was not computable"


def test_speed_is_expressed_in_body_lengths_per_second(tracklet, poses, profile, config) -> None:
    """The fixture walks 6 px per frame at 10 fps with an 80 px body."""
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert record.value("speed") == pytest.approx(60.0 / 80.0, rel=0.05)


def test_records_produced_under_different_feature_versions_are_distinguishable(
    tracklet, poses, profile, config
) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    first = FeatureExtractor(profile, config).extract(lane_pass, poses)

    bumped = dataclasses.replace(
        profile, feature_set=dataclasses.replace(profile.feature_set, version="1")
    )
    second = FeatureExtractor(bumped, config).extract(lane_pass, poses)
    assert first.feature_set_version != second.feature_set_version


# -- 6.4 undeclared features are refused ------------------------------------


def test_an_undeclared_feature_fails_extraction_naming_it(tracklet, poses, profile, config) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    extractor = FeatureExtractor(profile, config)
    original = extractor._compute

    def with_extra(tracks, seconds):
        values = original(tracks, seconds)
        values["hock_angle_variance"] = 0.42
        return values

    extractor._compute = with_extra
    with pytest.raises(UndeclaredFeatureError) as excinfo:
        extractor.extract(lane_pass, poses)
    assert excinfo.value.feature_name == "hock_angle_variance"
    assert "hock_angle_variance" in str(excinfo.value)


def test_the_undeclared_feature_is_not_emitted_at_all(tracklet, poses, profile, config) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert "hock_angle_variance" not in {f.name for f in record.features}


# -- 6.5 per-feature quality flags ------------------------------------------


def test_a_feature_from_an_intermittent_keypoint_is_flagged_and_names_it(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=40, intermittent={"head": 0.5})
    tracklet = tracklet_from_poses(poses)
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)

    head_feature = next(f for f in record.features if f.name == "head_lateral_offset")
    assert head_feature.quality is QualityFlag.REDUCED
    assert head_feature.limiting_keypoint == "head"
    assert "head" in head_feature.note

    # A feature that does not depend on the head is unaffected.
    sway = next(f for f in record.features if f.name == "lateral_sway")
    assert sway.quality is QualityFlag.GOOD


def test_a_feature_whose_keypoint_is_absent_is_unusable(profile, config) -> None:
    poses = synthetic_pose_sequence(
        profile,
        frames=40,
        visible=("withers", "base_of_tail", "neck", "left_front_paw", "right_front_paw"),
    )
    tracklet = tracklet_from_poses(poses)
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)

    head_feature = next(f for f in record.features if f.name == "head_lateral_offset")
    assert head_feature.quality is QualityFlag.UNUSABLE
    assert head_feature.limiting_keypoint == "head"


def test_quality_reflects_coverage_not_merely_presence(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=40, intermittent={"left_back_paw": 0.25})
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    feature = next(f for f in record.features if f.name == "stride_length_back")
    assert feature.coverage < 0.8
    assert feature.quality is not QualityFlag.GOOD


# -- 6.6 explicit pass validity ---------------------------------------------


def test_a_partial_pass_is_invalid_with_a_reason(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=12, speed=6.0)
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert not record.valid
    assert record.validity_reason is ValidityReason.PARTIAL_PASS


def test_a_pass_with_too_many_reduced_features_is_invalid(profile, config) -> None:
    poses = synthetic_pose_sequence(
        profile,
        frames=40,
        intermittent={
            "left_front_paw": 0.5,
            "right_front_paw": 0.5,
            "left_back_paw": 0.5,
            "right_back_paw": 0.5,
            "head": 0.5,
            "neck": 0.5,
        },
    )
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)
    assert not record.valid
    assert record.validity_reason is ValidityReason.TOO_MANY_REDUCED_FEATURES


def test_an_invalid_pass_presents_no_measurements_downstream(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=12, speed=6.0)
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)

    assert not record.valid
    assert record.measurements() == ()
    assert record.as_mapping() == {}


def test_an_invalid_pass_is_retained_with_its_reason_for_audit(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=12, speed=6.0)
    lane_pass = segment_passes(tracklet_from_poses(poses), config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(lane_pass, poses)

    restored = FeatureRecord.from_dict(record.to_dict())
    assert restored.validity_reason is ValidityReason.PARTIAL_PASS
    # The values are still there to inspect; they are simply not measurements.
    assert len(restored.features) == len(record.features)
    assert restored.measurements() == ()


def test_an_unresolved_identity_makes_the_pass_invalid(tracklet, poses, profile, config) -> None:
    lane_pass = segment_passes(tracklet, config, **FRAME)[0]
    record = FeatureExtractor(profile, config).extract(
        lane_pass, poses, animal_id="a1", identity_resolved=False
    )
    assert not record.valid
    assert record.validity_reason is ValidityReason.UNRESOLVED_IDENTITY
    assert record.animal_id == ""


def test_a_too_short_pass_is_invalid(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=6, speed=30.0)
    passes = segment_passes(tracklet_from_poses(poses), config, **FRAME)
    record = FeatureExtractor(profile, config).extract(passes[0], poses)
    assert not record.valid
    assert record.validity_reason in {
        ValidityReason.TOO_FEW_FRAMES,
        ValidityReason.PARTIAL_PASS,
    }


def test_accepting_partial_passes_is_a_declared_configuration_choice(profile, config) -> None:
    poses = synthetic_pose_sequence(profile, frames=14, speed=6.0)
    lenient = dataclasses.replace(
        config, phenotype=dataclasses.replace(config.phenotype, accept_partial_passes=True)
    )
    lane_pass = segment_passes(tracklet_from_poses(poses), lenient, **FRAME)[0]
    record = FeatureExtractor(profile, lenient).extract(lane_pass, poses)

    assert record.validity_reason is not ValidityReason.PARTIAL_PASS
    assert lenient.digest != config.digest, "the choice must be visible in the digest"
