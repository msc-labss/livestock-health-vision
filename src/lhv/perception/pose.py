"""Pose estimation against the profile's skeleton.

The stage's contract is the profile skeleton, not whatever convention a set of
weights happens to use. A backend declares its native convention and a mapping
onto the profile skeleton; every profile keypoint the backend cannot produce is
emitted not-visible with no coordinate.

That is the honest arrangement while no publicly downloadable checkpoint matches
the convention the profile declares exactly. It also means substituting such a
checkpoint later is a profile edit rather than a change here.

Before any of that runs, the source's declared view must match the view the
skeleton is defined for. A skeleton's view decides what its keypoints mean
spatially, so the pairing is checked once per source and refused by name rather
than producing coordinates whose interpretation nobody stated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import ResolvedConfig
from ..errors import ConfigError, ViewMismatchError
from ..profiles import SkeletonDefinition, SpeciesProfile
from .schemas import Detection, Keypoint, Pose, Visibility

__all__ = [
    "NativeKeypoint",
    "PoseBackend",
    "PoseEstimator",
    "AnnotationPoseBackend",
    "UltralyticsPoseBackend",
    "map_to_skeleton",
    "IMPLEMENTED_RUNTIMES",
]


@dataclass(frozen=True)
class NativeKeypoint:
    """A keypoint as a backend produced it, in its own naming convention."""

    name: str
    x: float
    y: float
    confidence: float


class PoseBackend(Protocol):
    @property
    def model_identity(self) -> str: ...

    @property
    def native_skeleton(self) -> str: ...

    @property
    def keypoint_map(self) -> dict[str, str]:
        """Native keypoint name -> profile skeleton keypoint name."""
        ...

    def estimate(self, image: np.ndarray, detection: Detection) -> list[NativeKeypoint]: ...


def map_to_skeleton(
    native: list[NativeKeypoint],
    *,
    skeleton: SkeletonDefinition,
    keypoint_map: dict[str, str],
    visibility_threshold: float,
) -> tuple[Keypoint, ...]:
    """Project a backend's output onto the profile skeleton.

    Anything the backend did not produce, or produced below the visibility
    threshold, becomes a not-visible keypoint with no coordinate. Nothing is
    interpolated, and no neighbour's position is borrowed.
    """
    by_profile_name: dict[str, NativeKeypoint] = {}
    for item in native:
        profile_name = keypoint_map.get(item.name, item.name)
        if profile_name not in skeleton.names:
            continue
        existing = by_profile_name.get(profile_name)
        if existing is None or item.confidence > existing.confidence:
            by_profile_name[profile_name] = item

    keypoints: list[Keypoint] = []
    for definition in sorted(skeleton.keypoints, key=lambda k: k.index):
        found = by_profile_name.get(definition.name)
        if found is None:
            keypoints.append(
                Keypoint(
                    name=definition.name,
                    x=None,
                    y=None,
                    confidence=0.0,
                    visibility=Visibility.NOT_VISIBLE,
                )
            )
            continue
        if found.confidence < visibility_threshold:
            keypoints.append(
                Keypoint(
                    name=definition.name,
                    x=None,
                    y=None,
                    confidence=float(found.confidence),
                    visibility=Visibility.NOT_VISIBLE,
                )
            )
            continue
        keypoints.append(
            Keypoint(
                name=definition.name,
                x=float(found.x),
                y=float(found.y),
                confidence=float(found.confidence),
                visibility=Visibility.VISIBLE,
            )
        )
    return tuple(keypoints)


class AnnotationPoseBackend:
    """Serves keypoints that were annotated rather than predicted.

    Used to evaluate the stages after pose without the placeholder pose model's
    error dominating, and to exercise the contract in tests. Its identity says
    plainly that the keypoints are labels, so no report can mistake them for a
    model's output.
    """

    def __init__(
        self,
        annotations: dict[str, list[NativeKeypoint]],
        *,
        source: str = "annotation",
        version: str = "1",
        native_skeleton: str = "profile",
        keypoint_map: dict[str, str] | None = None,
    ) -> None:
        self._annotations = annotations
        self._source = source
        self._version = version
        self._native_skeleton = native_skeleton
        self._keypoint_map = keypoint_map or {}

    @property
    def model_identity(self) -> str:
        return f"{self._source}@{self._version}"

    @property
    def native_skeleton(self) -> str:
        return self._native_skeleton

    @property
    def keypoint_map(self) -> dict[str, str]:
        return self._keypoint_map

    def estimate(self, image: np.ndarray, detection: Detection) -> list[NativeKeypoint]:
        return list(self._annotations.get(detection.detection_id, []))


class UltralyticsPoseBackend:
    """A pretrained pose model, cropped to a detection and mapped to the skeleton."""

    def __init__(
        self,
        weights: str,
        *,
        name: str,
        version: str,
        native_skeleton: str,
        keypoint_map: dict[str, str],
        device: str = "auto",
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._name = name
        self._version = version
        self._native_skeleton = native_skeleton
        self._keypoint_map = dict(keypoint_map)
        self._device = None if device == "auto" else device

    @property
    def model_identity(self) -> str:
        return f"{self._name}@{self._version}"

    @property
    def native_skeleton(self) -> str:
        return self._native_skeleton

    @property
    def keypoint_map(self) -> dict[str, str]:
        return self._keypoint_map

    def estimate(self, image: np.ndarray, detection: Detection) -> list[NativeKeypoint]:
        x1 = max(int(detection.box.x1), 0)
        y1 = max(int(detection.box.y1), 0)
        x2 = min(int(detection.box.x2), image.shape[1])
        y2 = min(int(detection.box.y2), image.shape[0])
        if x2 <= x1 or y2 <= y1:
            return []
        crop = image[y1:y2, x1:x2]

        predictions = self._model.predict(crop, device=self._device, verbose=False)
        results: list[NativeKeypoint] = []
        for prediction in predictions:
            keypoints = getattr(prediction, "keypoints", None)
            if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
                continue
            names = self._native_names(len(keypoints.xy[0]))
            coordinates = keypoints.xy[0].tolist()
            confidences = (
                keypoints.conf[0].tolist()
                if getattr(keypoints, "conf", None) is not None
                else [1.0] * len(coordinates)
            )
            for name, (x, y), confidence in zip(names, coordinates, confidences, strict=False):
                # Back into full-frame coordinates; downstream features are
                # computed in the frame, not in the crop.
                results.append(
                    NativeKeypoint(
                        name=name, x=float(x) + x1, y=float(y) + y1, confidence=float(confidence)
                    )
                )
            break
        return results

    def _native_names(self, count: int) -> list[str]:
        declared = list(self._keypoint_map)
        if len(declared) >= count:
            return declared[:count]
        return declared + [f"native_{i}" for i in range(len(declared), count)]


class PoseEstimator:
    """Runs a pose backend against the profile skeleton and counts what it produced."""

    def __init__(
        self,
        backend: PoseBackend,
        profile: SpeciesProfile,
        config: ResolvedConfig,
        *,
        source_id: str,
        source_view: str,
    ) -> None:
        skeleton_view = profile.skeleton.view
        if source_view != skeleton_view:
            # Refused here rather than per frame: the mismatch is a property of
            # the pairing, so failing once before any frame is read is both the
            # correct scope and the cheaper one.
            raise ViewMismatchError(
                source_id=source_id,
                source_view=source_view,
                skeleton_id=profile.skeleton.identifier,
                skeleton_view=skeleton_view,
            )
        self.backend = backend
        self.profile = profile
        self.config = config
        self.source_id = source_id
        self.view = source_view
        self.poses_emitted = 0
        self.low_confidence_poses = 0
        self.keypoints_emitted = 0
        self.keypoints_not_visible = 0

    @property
    def model_identity(self) -> str:
        return self.backend.model_identity

    def estimate(
        self,
        image: np.ndarray,
        detection: Detection,
        *,
        tracklet_id: str = "",
    ) -> Pose:
        native = self.backend.estimate(image, detection)
        keypoints = map_to_skeleton(
            native,
            skeleton=self.profile.skeleton,
            keypoint_map=self.backend.keypoint_map,
            visibility_threshold=self.config.perception.keypoint_visibility_threshold,
        )
        observed = [k for k in keypoints if k.observed]
        mean_confidence = sum(k.confidence for k in observed) / len(observed) if observed else 0.0
        low = mean_confidence < self.config.perception.pose_low_confidence_threshold

        self.poses_emitted += 1
        self.keypoints_emitted += len(keypoints)
        self.keypoints_not_visible += len(keypoints) - len(observed)
        if low:
            self.low_confidence_poses += 1

        return Pose(
            pose_id=f"{detection.detection_id}:pose",
            provenance=detection.provenance,
            detection_id=detection.detection_id,
            keypoints=keypoints,
            skeleton_id=self.profile.skeleton.identifier,
            skeleton_version=self.profile.skeleton.version,
            model_identity=self.model_identity,
            tracklet_id=tracklet_id,
            low_confidence=low,
            mean_confidence=mean_confidence,
            view=self.view,
        )


# Weight runtimes this module can actually load. A profile may name one that is
# not here — declaring an intended backend before it is wired is honest — and
# building it is refused by name rather than attempted.
IMPLEMENTED_RUNTIMES = ("ultralytics",)


def pose_backend_from_profile(
    profile: SpeciesProfile,
    config: ResolvedConfig,
    *,
    weights_path: str | None = None,
) -> PoseBackend:
    """Build the profile's declared pose backend.

    The native convention and its mapping onto the profile skeleton both come
    from the profile, so this function knows nothing about the animal.

    It does have to know which checkpoint formats it can read. A profile is free
    to declare weights for a runtime this module does not implement, and the
    refusal below is what keeps that declaration from becoming a crash inside a
    loader being handed a file it cannot parse.
    """
    reference = profile.weight("pose")
    if reference.runtime not in IMPLEMENTED_RUNTIMES:
        raise ConfigError(
            f"profile {profile.identifier} declares pose weights {reference.name!r} for the "
            f"{reference.runtime!r} runtime, which this pipeline does not implement "
            f"(implemented: {', '.join(IMPLEMENTED_RUNTIMES)}). Either implement that runtime "
            f"or point 'weights.pose' at a checkpoint one of them can load; the weights are a "
            f"profile edit and the runtime is not."
        )
    return UltralyticsPoseBackend(
        weights_path or reference.uri,
        name=reference.name,
        version=reference.version,
        native_skeleton=reference.native_skeleton,
        keypoint_map=reference.keypoint_map,
        device=config.perception.device,
    )
