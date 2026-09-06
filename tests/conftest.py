"""Shared fixtures.

Synthetic media stands in for the public datasets throughout the unit tests. It
is deliberately crude: a bright blob walking down a lane against a dark ground.
It exercises the contracts — provenance, determinism, tracking, segmentation —
without asserting anything about model accuracy, which only real labelled data
can measure.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from lhv.config import ModelIdentity, ResolvedConfig
from lhv.ingest import register_source
from lhv.profiles import load_profile

FIXED_START = datetime(2024, 3, 1, 6, 30, 0, tzinfo=UTC)


def render_lane_frame(
    width: int = 320,
    height: int = 240,
    *,
    progress: float,
    phase: float = 0.0,
    sway: float = 0.0,
    present: bool = True,
) -> np.ndarray:
    """One frame of a blob progressing down the lane, with a swinging gait."""
    import cv2

    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (width // 4, 0), (3 * width // 4, height), (60, 60, 60), -1)
    if not present:
        return image

    centre_y = int(progress * height)
    centre_x = int(width / 2 + sway * width * 0.05 * np.sin(phase))
    cv2.ellipse(image, (centre_x, centre_y), (28, 46), 0, 0, 360, (220, 220, 220), -1)
    # Limbs, so a pose-like structure exists to track.
    for side, offset in (("left", -22), ("right", 22)):
        swing = int(10 * np.sin(phase + (0.0 if side == "left" else np.pi)))
        cv2.circle(image, (centre_x + offset, centre_y - 26 + swing), 6, (150, 150, 150), -1)
        cv2.circle(image, (centre_x + offset, centre_y + 26 - swing), 6, (150, 150, 150), -1)
    return image


def write_video(
    path: Path,
    *,
    frames: int = 40,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
    absent_from: int | None = None,
) -> Path:
    """Write a synthetic lane video. Returns the path actually written."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    if not writer.isOpened():
        pytest.skip("no usable video writer in this OpenCV build")
    try:
        for index in range(frames):
            present = absent_from is None or index < absent_from
            writer.write(
                render_lane_frame(
                    width,
                    height,
                    progress=index / max(frames - 1, 1),
                    phase=index * 0.8,
                    sway=1.0,
                    present=present,
                )
            )
    finally:
        writer.release()
    return path


def write_image_sequence(directory: Path, *, frames: int = 12) -> Path:
    import cv2

    directory.mkdir(parents=True, exist_ok=True)
    for index in range(frames):
        cv2.imwrite(
            str(directory / f"frame_{index:04d}.jpg"),
            render_lane_frame(progress=index / max(frames - 1, 1), phase=index * 0.8),
        )
    return directory


@pytest.fixture
def config() -> ResolvedConfig:
    return ResolvedConfig(
        species_profile="cattle",
        species_profile_version="0",
        dataset_name="synthetic",
        dataset_version="1",
        models={
            "detector": ModelIdentity(name="stub-detector", version="1", task="detect"),
            "pose": ModelIdentity(name="stub-pose", version="1", task="pose"),
        },
    )


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    return write_video(tmp_path / "media" / "lane.avi")


@pytest.fixture
def source(video_path: Path):
    return register_source(
        source_id="synthetic/lane",
        camera_id="cam-lane-1",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(video_path),
        dataset_name="synthetic",
        dataset_version="1",
        start_timestamp=FIXED_START,
        kind="video",
        view=load_profile("cattle").skeleton.view,
    )


@pytest.fixture
def output_root(tmp_path: Path) -> Path:
    root = tmp_path / "runs"
    root.mkdir(parents=True, exist_ok=True)
    return root


def synthetic_pose_sequence(
    profile,
    *,
    frames: int = 24,
    fps: float = 10.0,
    start_index: int = 0,
    stride_amplitude: float = 12.0,
    left_amplitude: float | None = None,
    right_amplitude: float | None = None,
    sway: float = 2.0,
    speed: float = 6.0,
    visible: tuple[str, ...] | None = None,
    intermittent: dict[str, float] | None = None,
    source_id: str = "s",
    site_key: str = "site-a",
    day_key: str = "2024-03-01",
    reliable: bool = True,
):
    """Poses of one animal walking a lane, with a controllable gait.

    Coordinates are laid out so that the body axis runs along +y, the limbs swing
    along the axis, and the body sways across it — which is what the feature set
    measures.
    """
    from datetime import timedelta

    import numpy as np

    from lhv.ingest import FrameProvenance
    from lhv.perception import Keypoint, Pose, Visibility

    left_amplitude = stride_amplitude if left_amplitude is None else left_amplitude
    right_amplitude = stride_amplitude if right_amplitude is None else right_amplitude
    visible = visible or (
        "withers",
        "sacrum",
        "nose",
        "left_front_hoof",
        "right_front_hoof",
        "left_hind_hoof",
        "right_hind_hoof",
    )
    intermittent = intermittent or {}

    poses = []
    for step in range(frames):
        index = start_index + step
        phase = 0.9 * step
        y = 20.0 + speed * step
        x = 160.0 + sway * np.sin(0.5 * step)

        # A lateral view: the animal travels along y, and the axis perpendicular
        # to travel in the image is the sagittal vertical. ``sway`` therefore
        # drives vertical excursion here, which is what head bob is measured on.
        positions = {
            "withers": (x, y - 40.0),
            "sacrum": (x, y + 40.0),
            # forehead and mid_thoracic are deliberately absent: the AP-10K
            # backend does not emit them, and back_posture is declared
            # unavailable for exactly that reason.
            "nose": (x + 0.35 * sway * np.sin(0.9 * step), y - 70.0),
            "left_front_hoof": (x - 20.0, y - 30.0 + left_amplitude * np.sin(phase)),
            "right_front_hoof": (x + 20.0, y - 30.0 + right_amplitude * np.sin(phase + np.pi)),
            "left_hind_hoof": (x - 20.0, y + 30.0 + left_amplitude * np.sin(phase + np.pi)),
            "right_hind_hoof": (x + 20.0, y + 30.0 + right_amplitude * np.sin(phase)),
        }

        keypoints = []
        for name in profile.skeleton.names:
            drop = intermittent.get(name)
            present = name in visible and name in positions
            if present and drop is not None and (step % max(int(1 / drop), 1)) != 0:
                present = False
            if not present:
                keypoints.append(
                    Keypoint(
                        name=name,
                        x=None,
                        y=None,
                        confidence=0.0,
                        visibility=Visibility.NOT_VISIBLE,
                    )
                )
                continue
            px, py = positions[name]
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(px),
                    y=float(py),
                    confidence=0.9,
                    visibility=Visibility.VISIBLE,
                )
            )

        provenance = FrameProvenance(
            source_id=source_id,
            camera_id="cam-1",
            frame_index=index,
            site_key=site_key,
            animal_set_key="herd-a",
            day_key=day_key,
            capture_timestamp=(FIXED_START + timedelta(seconds=step / fps)) if reliable else None,
            timestamp_reliable=reliable,
        )
        poses.append(
            Pose(
                pose_id=f"{source_id}#{index}:pose",
                provenance=provenance,
                detection_id=f"{source_id}#{index}:0",
                keypoints=tuple(keypoints),
                skeleton_id=profile.skeleton.identifier,
                skeleton_version=profile.skeleton.version,
                model_identity="annotation@1",
                mean_confidence=0.9,
            )
        )
    return poses


def tracklet_from_poses(poses, *, tracklet_id: str = "s:t00001", box_half: float = 45.0):
    """A tracklet whose detections follow the poses' body midpoints."""
    from lhv.perception import BoundingBox, Detection, TerminationReason, Tracklet

    detections = []
    for pose in poses:
        withers = pose.keypoint("withers")
        tail = pose.keypoint("sacrum")
        cx = (withers.x + tail.x) / 2.0
        cy = (withers.y + tail.y) / 2.0
        detections.append(
            Detection(
                detection_id=pose.detection_id,
                provenance=pose.provenance,
                box=BoundingBox(cx - box_half, cy - box_half, cx + box_half, cy + box_half),
                label="animal",
                confidence=0.9,
                model_identity="stub@1",
            )
        )
    first, last = detections[0], detections[-1]
    reliable = all(d.provenance.timestamp_reliable for d in detections)
    return Tracklet(
        tracklet_id=tracklet_id,
        source_id=first.provenance.source_id,
        camera_id=first.provenance.camera_id,
        site_key=first.provenance.site_key,
        animal_set_key=first.provenance.animal_set_key,
        day_key=first.provenance.day_key,
        first_frame_index=first.frame_index,
        last_frame_index=last.frame_index,
        detections=tuple(detections),
        termination_reason=TerminationReason.EXIT,
        model_identity="stub@1",
        first_timestamp=first.provenance.capture_timestamp if reliable else None,
        last_timestamp=last.provenance.capture_timestamp if reliable else None,
        timestamps_reliable=reliable,
    )


def feature_record(
    *,
    animal_id: str,
    observed_at,
    pass_id: str,
    values: dict[str, float],
    profile,
    site_key: str = "site-a",
    valid: bool = True,
    injected: bool = False,
):
    """A feature record carrying exactly the named values, all of good quality."""
    from lhv.phenotype import (
        FeatureRecord,
        FeatureValue,
        PassCompleteness,
        QualityFlag,
        ValidityReason,
    )

    features = []
    for name, value in values.items():
        definition = profile.feature_set.get(name)
        assert definition is not None, f"{name} is not in the feature set"
        features.append(
            FeatureValue(name=name, value=value, unit=definition.unit, quality=QualityFlag.GOOD)
        )
    return FeatureRecord(
        pass_id=pass_id,
        feature_set_version=profile.feature_set.version,
        skeleton_id=profile.skeleton.identifier,
        skeleton_version=profile.skeleton.version,
        site_key=site_key,
        day_key=observed_at.date().isoformat(),
        valid=valid,
        validity_reason=ValidityReason.VALID if valid else ValidityReason.TOO_FEW_FRAMES,
        features=tuple(features),
        tracklet_id=f"{pass_id}:t",
        animal_id=animal_id,
        observed_at=observed_at,
        completeness=PassCompleteness.COMPLETE,
        injected=injected,
    )


def render_pose_frame(pose, *, width: int = 320, height: int = 240):
    """Render a frame that actually agrees with the pose labels drawn on it.

    The pixels and the labels come from the same coordinates, so a detector run
    over the video and a pose backend served from the labels are describing the
    same animal rather than two loosely related ones.
    """
    import cv2
    import numpy as np

    image = np.full((height, width, 3), 30, dtype=np.uint8)
    cv2.rectangle(image, (width // 4, 0), (3 * width // 4, height), (60, 60, 60), -1)

    withers = pose.keypoint("withers")
    tail = pose.keypoint("sacrum")
    if withers is None or tail is None or not withers.observed or not tail.observed:
        return image

    cx = int((withers.x + tail.x) / 2)
    cy = int((withers.y + tail.y) / 2)
    half = int(abs(tail.y - withers.y) / 2) + 6
    cv2.ellipse(image, (cx, cy), (28, half), 0, 0, 360, (225, 225, 225), -1)
    for keypoint in pose.keypoints:
        if keypoint.observed and keypoint.name.endswith("paw"):
            cv2.circle(image, (int(keypoint.x), int(keypoint.y)), 6, (170, 170, 170), -1)
    return image


def write_labelled_lane_dataset(
    root,
    profile,
    *,
    animals: int = 4,
    days: int = 8,
    frames: int = 40,
    fps: float = 10.0,
    width: int = 320,
    height: int = 240,
):
    """A small labelled dataset shaped like the public sources.

    One video per animal per day, with the keypoint labels that produced it and
    an identity label per sequence — the same shape as a top-down keypoint
    dataset with tracking identifiers, at a size a test can run.
    """
    import json
    from datetime import timedelta
    from pathlib import Path

    import cv2

    root = Path(root)
    labels: dict[str, dict] = {}

    for animal in range(animals):
        animal_id = f"animal-{animal:02d}"
        for day in range(days):
            day_key = (FIXED_START + timedelta(days=day)).date().isoformat()
            directory = root / day_key
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{animal_id}.avi"

            poses = synthetic_pose_sequence(
                profile,
                frames=frames,
                fps=fps,
                # Animals differ from each other; a given animal differs a
                # little from day to day. Both are needed for a baseline to
                # have any spread to measure against.
                speed=6.0 + 0.35 * animal + 0.05 * ((day * 7 + animal) % 5),
                left_amplitude=12.0 + 0.8 * animal + 0.15 * ((day * 3 + animal) % 4),
                right_amplitude=12.0 + 0.8 * animal + 0.15 * ((day * 5 + animal) % 3),
                sway=2.0 + 0.2 * animal,
                source_id=f"{day_key}/{animal_id}.avi",
                day_key=day_key,
            )

            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
            )
            if not writer.isOpened():
                pytest.skip("no usable video writer in this OpenCV build")
            try:
                for pose in poses:
                    writer.write(render_pose_frame(pose, width=width, height=height))
            finally:
                writer.release()

            labels[str(path.relative_to(root))] = {
                "animal_id": animal_id,
                "day_key": day_key,
                # Each animal passes at its own time of day, so an anchor window
                # identifies exactly one of them.
                "start_offset_seconds": animal * 600,
                "frames": [
                    {
                        "frame_index": pose.provenance.frame_index,
                        "keypoints": {k.name: [k.x, k.y] for k in pose.keypoints if k.observed},
                    }
                    for pose in poses
                ],
            }

    (root / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    return root


class LabelPoseBackend:
    """Serves the dataset's keypoint labels, keyed the way a label file is.

    A real dataset adapter looks like this: labels are addressed by source and
    frame, not by an identifier the pipeline invented at run time.
    """

    def __init__(self, labels: dict, *, version: str = "1") -> None:
        self._by_source_frame: dict[tuple[str, int], dict] = {}
        for relative, entry in labels.items():
            for frame in entry["frames"]:
                self._by_source_frame[(relative, frame["frame_index"])] = frame["keypoints"]
        self._version = version

    @property
    def model_identity(self) -> str:
        return f"dataset-keypoint-label@{self._version}"

    @property
    def native_skeleton(self) -> str:
        return "profile"

    @property
    def keypoint_map(self) -> dict:
        return {}

    def estimate(self, image, detection):
        from lhv.perception import NativeKeypoint

        source_id = detection.provenance.source_id
        relative = source_id.split("/", 1)[1] if "/" in source_id else source_id
        keypoints = self._by_source_frame.get((relative, detection.provenance.frame_index), {})
        return [
            NativeKeypoint(name=name, x=x, y=y, confidence=0.95)
            for name, (x, y) in keypoints.items()
        ]


def anchors_from_labels(labels: dict, *, site_key: str, dataset_name: str):
    """Route the dataset's identity labels through the external-anchor interface."""
    from datetime import timedelta

    from lhv.identity import AnchorRecord, DatasetLabelAnchorSource

    records = []
    for entry in labels.values():
        frames = entry["frames"]
        start = FIXED_START + timedelta(
            days=(datetime.fromisoformat(entry["day_key"]).date() - FIXED_START.date()).days,
            seconds=entry["start_offset_seconds"],
        )
        records.append(
            AnchorRecord(
                animal_id=entry["animal_id"],
                anchor_source=f"dataset-label:{dataset_name}",
                site_key=site_key,
                observed_from=start,
                observed_to=start + timedelta(seconds=len(frames) / 10.0 + 1),
                reader_id="ground-truth",
            )
        )
    return DatasetLabelAnchorSource(records, dataset_name=dataset_name)
