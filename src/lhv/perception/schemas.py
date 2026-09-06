"""Record schemas at the perception stage boundaries.

Detection, tracklet and pose are three separate durable contracts rather than
one, because they are re-run independently and because a breaking change to one
should not silently invalidate the others. Each carries its own version and the
identity of the model that produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from ..ingest.provenance import FrameProvenance
from ..schema import Record, opt, req

__all__ = [
    "BoundingBox",
    "Visibility",
    "TerminationReason",
    "Detection",
    "FrameDetections",
    "Keypoint",
    "Pose",
    "Tracklet",
]


@dataclass(frozen=True)
class BoundingBox:
    """An axis-aligned region in pixel coordinates of the source frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(self.x2 - self.x1, 0.0)

    @property
    def height(self) -> float:
        return max(self.y2 - self.y1, 0.0)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def iou(self, other: BoundingBox) -> float:
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)
        if right <= left or bottom <= top:
            return 0.0
        overlap = (right - left) * (bottom - top)
        union = self.area + other.area - overlap
        return overlap / union if union > 0 else 0.0

    def translated(self, dx: float, dy: float) -> BoundingBox:
        return BoundingBox(self.x1 + dx, self.y1 + dy, self.x2 + dx, self.y2 + dy)

    def touches_border(self, width: int, height: int, *, margin: float = 4.0) -> bool:
        return (
            self.x1 <= margin
            or self.y1 <= margin
            or self.x2 >= width - margin
            or self.y2 >= height - margin
        )


class Visibility(StrEnum):
    """Per-keypoint visibility.

    ``OCCLUDED`` means the annotator or model located a keypoint it could not
    see; ``NOT_VISIBLE`` means no observation exists. A NOT_VISIBLE keypoint
    carries no coordinate, so an interpolated position can never be mistaken for
    a measurement.
    """

    VISIBLE = "visible"
    OCCLUDED = "occluded"
    NOT_VISIBLE = "not_visible"


class TerminationReason(StrEnum):
    """Why a tracklet ended. Three distinct failures plus the benign one."""

    EXIT = "exit"
    OCCLUSION_LOSS = "occlusion_loss"
    DETECTION_FAILURE = "detection_failure"
    SOURCE_END = "source_end"


@dataclass(frozen=True)
class Detection(Record):
    SCHEMA_NAME = "detection"
    SCHEMA_VERSION = "1"

    detection_id: str = req()
    provenance: FrameProvenance = req()
    box: BoundingBox = req()
    label: str = req()
    confidence: float = req()
    model_identity: str = req()
    low_confidence: bool = opt(False)

    @property
    def frame_index(self) -> int:
        return self.provenance.frame_index


@dataclass(frozen=True)
class FrameDetections(Record):
    """Detection result for exactly one frame.

    A frame containing no animal produces this record with an empty tuple. The
    frame is never simply absent from the stream, because absence and "nothing
    there" are different facts and downstream counting depends on the
    difference.
    """

    SCHEMA_NAME = "frame_detections"
    SCHEMA_VERSION = "1"

    provenance: FrameProvenance = req()
    model_identity: str = req()
    detections: tuple[Detection, ...] = opt(())
    frame_width: int = opt(0)
    frame_height: int = opt(0)

    @property
    def is_empty(self) -> bool:
        return len(self.detections) == 0

    def __len__(self) -> int:
        return len(self.detections)


@dataclass(frozen=True)
class Keypoint:
    """One named keypoint. ``x`` and ``y`` are None whenever nothing was observed."""

    name: str
    x: float | None
    y: float | None
    confidence: float
    visibility: Visibility

    @property
    def observed(self) -> bool:
        return self.visibility is not Visibility.NOT_VISIBLE and self.x is not None

    def __post_init__(self) -> None:
        if self.visibility is Visibility.NOT_VISIBLE and (self.x is not None or self.y is not None):
            raise ValueError(
                f"keypoint {self.name!r} is not visible but carries a coordinate; a position "
                f"that was not observed must not be presented as one"
            )


@dataclass(frozen=True)
class Pose(Record):
    SCHEMA_NAME = "pose"
    # 2: added view, the camera geometry the keypoints were produced under.
    SCHEMA_VERSION = "2"

    pose_id: str = req()
    provenance: FrameProvenance = req()
    detection_id: str = req()
    keypoints: tuple[Keypoint, ...] = req()
    skeleton_id: str = req()
    skeleton_version: str = req()
    model_identity: str = req()
    tracklet_id: str = opt("")
    low_confidence: bool = opt(False)
    mean_confidence: float = opt(0.0)
    # The camera geometry these keypoints were produced under. Checked against
    # the skeleton's own view before any pose is emitted, and recorded here so a
    # stored pose says which geometry it means.
    view: str = opt("")

    def keypoint(self, name: str) -> Keypoint | None:
        for keypoint in self.keypoints:
            if keypoint.name == name:
                return keypoint
        return None

    @property
    def observed_names(self) -> tuple[str, ...]:
        return tuple(k.name for k in self.keypoints if k.observed)


@dataclass(frozen=True)
class Tracklet(Record):
    SCHEMA_NAME = "tracklet"
    SCHEMA_VERSION = "1"

    tracklet_id: str = req()
    source_id: str = req()
    camera_id: str = req()
    site_key: str = req()
    animal_set_key: str = req()
    day_key: str = req()
    first_frame_index: int = req()
    last_frame_index: int = req()
    detections: tuple[Detection, ...] = req()
    termination_reason: TerminationReason = req()
    model_identity: str = req()
    first_timestamp: datetime | None = opt(None)
    last_timestamp: datetime | None = opt(None)
    timestamps_reliable: bool = opt(True)

    @property
    def length(self) -> int:
        return len(self.detections)

    @property
    def frame_indices(self) -> tuple[int, ...]:
        return tuple(d.frame_index for d in self.detections)

    def overlaps_in_time(self, other: Tracklet) -> bool:
        """Whether two tracklets from the same source were observed at once."""
        if self.source_id != other.source_id:
            return False
        return (
            self.first_frame_index <= other.last_frame_index
            and other.first_frame_index <= self.last_frame_index
        )
