"""The detection stage.

Two backends share one interface: a pretrained neural detector whose weights the
species profile names, and a classical intensity-blob detector that needs no
weights at all. Both record their identity on every output, so a run's results
can never be attributed to the wrong one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..config import ResolvedConfig
from ..ingest.stream import Frame
from ..profiles import SpeciesProfile
from .schemas import BoundingBox, Detection, FrameDetections

__all__ = ["Detector", "DetectorBackend", "IntensityBlobDetector", "UltralyticsDetector"]


@dataclass
class _RawDetection:
    box: BoundingBox
    label: str
    confidence: float


class DetectorBackend(Protocol):
    @property
    def model_identity(self) -> str: ...

    def detect(self, image: np.ndarray) -> list[_RawDetection]: ...


class IntensityBlobDetector:
    """A weights-free detector over bright connected regions.

    It exists so the pipeline can be exercised end to end without downloading
    anything, and so continuous integration has a detector it can run. It is a
    measuring instrument of last resort, not a model: its identity says so, and
    no evaluation result produced with it may be presented as a detector result.
    """

    def __init__(
        self,
        *,
        threshold: int = 160,
        min_area: int = 200,
        label: str = "animal",
        version: str = "1",
    ) -> None:
        self.threshold = threshold
        self.min_area = min_area
        self.label = label
        self.version = version

    @property
    def model_identity(self) -> str:
        return f"intensity-blob@{self.version}"

    def detect(self, image: np.ndarray) -> list[_RawDetection]:
        import cv2

        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, mask = cv2.threshold(grey, self.threshold, 255, cv2.THRESH_BINARY)
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        results: list[_RawDetection] = []
        area_reference = float(grey.shape[0] * grey.shape[1])
        for index in range(1, count):
            x, y, width, height, area = stats[index]
            if area < self.min_area:
                continue
            results.append(
                _RawDetection(
                    box=BoundingBox(float(x), float(y), float(x + width), float(y + height)),
                    label=self.label,
                    # Confidence is the region's share of the frame, bounded to
                    # (0, 1). It is a proxy, and is named as one.
                    confidence=float(min(area / (area_reference * 0.08), 1.0)),
                )
            )
        results.sort(key=lambda d: (-d.confidence, d.box.x1, d.box.y1))
        return results


class UltralyticsDetector:
    """A pretrained detector whose weights and target classes the profile names."""

    def __init__(
        self,
        weights: str,
        *,
        name: str,
        version: str,
        target_classes: tuple[str, ...],
        device: str = "auto",
        confidence_threshold: float = 0.25,
    ) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._name = name
        self._version = version
        self._target_classes = tuple(c.lower() for c in target_classes)
        self._device = None if device == "auto" else device
        self._confidence_threshold = confidence_threshold

    @property
    def model_identity(self) -> str:
        return f"{self._name}@{self._version}"

    def detect(self, image: np.ndarray) -> list[_RawDetection]:
        predictions = self._model.predict(
            image,
            conf=self._confidence_threshold,
            device=self._device,
            verbose=False,
        )
        results: list[_RawDetection] = []
        for prediction in predictions:
            names = prediction.names
            for box in prediction.boxes:
                label = str(names[int(box.cls)]).lower()
                if self._target_classes and label not in self._target_classes:
                    continue
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                results.append(
                    _RawDetection(
                        box=BoundingBox(x1, y1, x2, y2),
                        label=label,
                        confidence=float(box.conf),
                    )
                )
        results.sort(key=lambda d: (-d.confidence, d.box.x1, d.box.y1))
        return results


class Detector:
    """Runs a backend over frames and emits one record per frame.

    Every frame produces a record, including a frame in which nothing was found.
    A detection below the operating threshold is emitted marked rather than
    dropped, and both facts are counted for the run summary.
    """

    def __init__(self, backend: DetectorBackend, config: ResolvedConfig) -> None:
        self.backend = backend
        self.config = config
        self.detections_emitted = 0
        self.low_confidence_detections = 0
        self.frames_processed = 0
        self.frames_without_detection = 0

    @property
    def model_identity(self) -> str:
        return self.backend.model_identity

    def detect_frame(self, frame: Frame) -> FrameDetections:
        if frame.image is None:
            raise ValueError(
                f"frame {frame.frame_id} carries no pixels; detection needs a decoded frame"
            )
        height, width = frame.image.shape[:2]
        raw = self.backend.detect(frame.image)

        threshold = self.config.perception.detection_threshold
        low_threshold = self.config.perception.detection_low_confidence_threshold

        detections: list[Detection] = []
        for position, item in enumerate(raw):
            if item.confidence < threshold:
                continue
            low = item.confidence < low_threshold
            detections.append(
                Detection(
                    detection_id=f"{frame.provenance.source_id}#{frame.index}:{position}",
                    provenance=frame.provenance,
                    box=item.box,
                    label=item.label,
                    confidence=item.confidence,
                    model_identity=self.model_identity,
                    low_confidence=low,
                )
            )
            if low:
                self.low_confidence_detections += 1

        self.frames_processed += 1
        self.detections_emitted += len(detections)
        if not detections:
            self.frames_without_detection += 1

        return FrameDetections(
            provenance=frame.provenance,
            model_identity=self.model_identity,
            detections=tuple(detections),
            frame_width=width,
            frame_height=height,
        )

    def detect_all(self, frames) -> list[FrameDetections]:
        return [self.detect_frame(frame) for frame in frames]


def detector_from_profile(
    profile: SpeciesProfile,
    config: ResolvedConfig,
    *,
    weights_path: str | None = None,
) -> Detector:
    """Build the profile's declared detector. Target classes come from the profile."""
    reference = profile.weight("detector")
    backend = UltralyticsDetector(
        weights_path or reference.uri,
        name=reference.name,
        version=reference.version,
        target_classes=reference.target_classes,
        device=config.perception.device,
        confidence_threshold=config.perception.detection_threshold,
    )
    return Detector(backend, config)
