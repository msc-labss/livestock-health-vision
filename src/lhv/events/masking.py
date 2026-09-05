"""Human masking.

Barn cameras capture workers and visitors, who are identifiable natural persons
even though the animals are not. Masking is applied in memory before any clip is
written, so there is never an unmasked copy on disk to delete later.

A masker that cannot do its job raises. The caller's response is to refuse the
clip, not to write it unmasked.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

__all__ = ["MaskingError", "Masker", "RegionMasker", "DetectorMasker"]


class MaskingError(RuntimeError):
    """Masking could not be applied, so nothing may be retained."""


class Masker(Protocol):
    @property
    def mode(self) -> str: ...

    def mask(self, image: np.ndarray) -> np.ndarray: ...


def _blur(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    import cv2

    x1, y1, x2, y2 = box
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])
    if x2 <= x1 or y2 <= y1:
        return image
    region = image[y1:y2, x1:x2]
    kernel = max(3, (min(region.shape[:2]) // 2) * 2 + 1)
    image[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
    return image


class RegionMasker:
    """Masks declared regions of the frame — a walkway, a doorway, a gate.

    Static regions need no model and no inference budget, which makes them the
    dependable floor beneath any detector-based masking.
    """

    def __init__(self, regions: list[tuple[float, float, float, float]], *, mode: str = "blur"):
        self.regions = list(regions)
        self._mode = mode

    @property
    def mode(self) -> str:
        return f"region-{self._mode}"

    def mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise MaskingError("cannot mask an empty frame")
        height, width = image.shape[:2]
        output = image.copy()
        for x1, y1, x2, y2 in self.regions:
            output = _blur(
                output,
                (int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)),
            )
        return output


class DetectorMasker:
    """Masks whatever a detector backend reports under the configured labels."""

    def __init__(self, backend, *, labels: tuple[str, ...] = ("person",), mode: str = "blur"):
        self.backend = backend
        self.labels = tuple(label.lower() for label in labels)
        self._mode = mode

    @property
    def mode(self) -> str:
        return f"detector-{self._mode}:{self.backend.model_identity}"

    def mask(self, image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            raise MaskingError("cannot mask an empty frame")
        try:
            detections = self.backend.detect(image, None)
        except Exception as exc:  # the detector is the masker's only evidence
            raise MaskingError(
                f"human detection failed, so masking cannot be applied: {exc}"
            ) from exc
        output = image.copy()
        for detection in detections:
            if detection.label.lower() not in self.labels:
                continue
            box = detection.box
            output = _blur(output, (int(box.x1), int(box.y1), int(box.x2), int(box.y2)))
        return output
