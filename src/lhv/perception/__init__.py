"""Stage boundary: frames -> detections, tracklets and poses.

Detection, motion-based tracking and keypoint pose estimation behind stable
interfaces, so pretrained weights are swappable without altering any downstream
contract.
"""

from .detect import Detector, IntensityBlobDetector, UltralyticsDetector, detector_from_profile
from .pose import (
    AnnotationPoseBackend,
    NativeKeypoint,
    PoseEstimator,
    UltralyticsPoseBackend,
    map_to_skeleton,
    pose_backend_from_profile,
)
from .report import PerceptionReport
from .schemas import (
    BoundingBox,
    Detection,
    FrameDetections,
    Keypoint,
    Pose,
    TerminationReason,
    Tracklet,
    Visibility,
)
from .track import Tracker, TrackerReport

__all__ = [
    "AnnotationPoseBackend",
    "BoundingBox",
    "Detection",
    "Detector",
    "FrameDetections",
    "IntensityBlobDetector",
    "Keypoint",
    "NativeKeypoint",
    "PerceptionReport",
    "Pose",
    "PoseEstimator",
    "TerminationReason",
    "Tracker",
    "TrackerReport",
    "Tracklet",
    "UltralyticsDetector",
    "UltralyticsPoseBackend",
    "Visibility",
    "detector_from_profile",
    "map_to_skeleton",
    "pose_backend_from_profile",
]
