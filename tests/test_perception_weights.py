"""Perception against the actual pretrained weights the profile names.

Marked ``gpu`` because it downloads weights and runs a network. Continuous
integration runs everything else; this is what makes the claim "the stage runs
against pretrained weights" checkable on a machine that has them.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lhv.config import ModelIdentity, ResolvedConfig
from lhv.ingest import Frame, FrameProvenance
from lhv.perception import (
    BoundingBox,
    Detection,
    Detector,
    PoseEstimator,
    UltralyticsDetector,
    Visibility,
    pose_backend_from_profile,
)
from lhv.profiles import load_profile

from .conftest import render_lane_frame

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"

pytestmark = pytest.mark.gpu


def _weights(name: str) -> str:
    path = WEIGHTS / name
    if not path.exists():
        pytest.skip(f"{path} not present; run tools/fetch_weights.py")
    return str(path)


@pytest.fixture
def profile():
    return load_profile("cattle")


@pytest.fixture
def weights_config(profile):
    return ResolvedConfig(
        species_profile=profile.species,
        species_profile_version=profile.version,
        dataset_name="synthetic",
        dataset_version="1",
        models={
            "detector": ModelIdentity(
                name=profile.weight("detector").name,
                version=profile.weight("detector").version,
                task="detect",
            )
        },
    )


def _provenance(index: int = 0) -> FrameProvenance:
    return FrameProvenance(
        source_id="s",
        camera_id="c",
        frame_index=index,
        site_key="site-a",
        animal_set_key="herd-a",
        day_key="2024-03-01",
        capture_timestamp=datetime(2024, 3, 1, tzinfo=UTC),
    )


def test_pretrained_detector_runs_and_records_its_identity(profile, weights_config) -> None:
    reference = profile.weight("detector")
    backend = UltralyticsDetector(
        _weights("yolo11m.pt"),
        name=reference.name,
        version=reference.version,
        target_classes=reference.target_classes,
        device="cpu",
        confidence_threshold=weights_config.perception.detection_threshold,
    )
    detector = Detector(backend, weights_config)
    record = detector.detect_frame(
        Frame(provenance=_provenance(), image=render_lane_frame(progress=0.5))
    )
    assert record.model_identity == f"{reference.name}@{reference.version}"
    assert detector.frames_processed == 1


def test_pretrained_detector_emits_an_empty_result_for_a_frame_with_no_animal(
    profile, weights_config
) -> None:
    reference = profile.weight("detector")
    backend = UltralyticsDetector(
        _weights("yolo11m.pt"),
        name=reference.name,
        version=reference.version,
        target_classes=reference.target_classes,
        device="cpu",
    )
    detector = Detector(backend, weights_config)
    blank = np.full((240, 320, 3), 20, dtype=np.uint8)
    record = detector.detect_frame(Frame(provenance=_provenance(1), image=blank))
    assert record.is_empty
    assert record.provenance.frame_index == 1
    assert detector.frames_without_detection == 1


def test_pretrained_pose_backend_emits_the_profile_skeleton(profile, weights_config) -> None:
    backend = pose_backend_from_profile(
        profile, weights_config, weights_path=_weights("yolo11m-pose.pt")
    )
    estimator = PoseEstimator(backend, profile, weights_config)
    detection = Detection(
        detection_id="d0",
        provenance=_provenance(),
        box=BoundingBox(120, 80, 200, 180),
        label="animal",
        confidence=0.9,
        model_identity="stub@1",
    )
    pose = estimator.estimate(render_lane_frame(progress=0.5), detection)

    assert [k.name for k in pose.keypoints] == list(profile.skeleton.names)
    assert pose.skeleton_id == profile.skeleton.identifier
    assert pose.skeleton_version == profile.skeleton.version
    assert pose.model_identity == f"{backend._name}@{backend._version}"
    # Whatever the model did or did not find, nothing unobserved carries a
    # coordinate.
    for keypoint in pose.keypoints:
        if keypoint.visibility is Visibility.NOT_VISIBLE:
            assert keypoint.x is None and keypoint.y is None
