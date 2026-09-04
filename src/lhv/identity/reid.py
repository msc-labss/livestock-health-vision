"""Visual re-identification — the fallback path only.

Appearance embeddings live here and nowhere else. Tracking within a pass uses
motion alone, so the expensive path stays optional and the edge phase is not
committed to carrying it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

__all__ = [
    "EmbeddingBackend",
    "ColourHistogramEmbedding",
    "GalleryMatch",
    "ReferenceGallery",
    "cosine_similarity",
]


class EmbeddingBackend(Protocol):
    @property
    def model_identity(self) -> str: ...

    def embed(self, image: np.ndarray) -> np.ndarray: ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    # Map from [-1, 1] to [0, 1] so a similarity can be read as a confidence.
    return float((np.dot(a, b) / denominator + 1.0) / 2.0)


@dataclass(frozen=True)
class GalleryMatch:
    """The closest enrolled animal, and how clearly it beat the next one.

    ``similarity`` is how alike the query and the reference are; ``separation``
    is how much better that was than the runner-up. Measured on
    MultiCamCows2024, absolute similarity carries almost no information about
    whether the match is right — correct and incorrect matches average 0.9965
    and 0.9964, an AUROC of 0.527 — while the separation does discriminate and
    yields a usable precision-recall trade. A floor placed on similarity alone
    therefore filters nothing; a floor on separation filters.
    """

    animal_id: str
    similarity: float
    separation: float = 0.0
    runner_up: str = ""


class ColourHistogramEmbedding:
    """A coat-pattern embedding from colour histograms.

    Deliberately simple and weights-free. Coat pattern is what the public
    re-identification data keys on, so a histogram is a defensible baseline for
    the fallback path — and being a baseline is exactly what it is reported as.
    """

    def __init__(self, *, bins: int = 8, version: str = "1") -> None:
        self.bins = bins
        self.version = version

    @property
    def model_identity(self) -> str:
        return f"colour-histogram@{self.version}"

    def embed(self, image: np.ndarray) -> np.ndarray:
        import cv2

        if image.size == 0:
            return np.zeros(self.bins**3, dtype=np.float32)
        histogram = cv2.calcHist([image], [0, 1, 2], None, [self.bins] * 3, [0, 256] * 3).flatten()
        total = histogram.sum()
        return (histogram / total).astype(np.float32) if total else histogram.astype(np.float32)


class ReferenceGallery:
    """Known animals and their reference embeddings."""

    def __init__(self, *, backend: EmbeddingBackend) -> None:
        self.backend = backend
        self._references: dict[str, list[np.ndarray]] = {}

    def enrol(self, animal_id: str, embedding: np.ndarray) -> None:
        self._references.setdefault(animal_id, []).append(np.asarray(embedding, dtype=np.float32))

    def enrol_image(self, animal_id: str, image: np.ndarray) -> None:
        self.enrol(animal_id, self.backend.embed(image))

    @property
    def animal_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._references))

    def __len__(self) -> int:
        return len(self._references)

    def rank(self, embedding: np.ndarray) -> list[tuple[str, float]]:
        """Every enrolled animal and its best similarity, most alike first."""
        scored = [
            (animal_id, max(cosine_similarity(embedding, r) for r in references))
            for animal_id, references in sorted(self._references.items())
            if references
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    def best_match(self, embedding: np.ndarray) -> GalleryMatch | None:
        """The closest enrolled animal, with its margin over the runner-up."""
        ranked = self.rank(embedding)
        if not ranked:
            return None
        animal_id, similarity = ranked[0]
        runner_up, second = ranked[1] if len(ranked) > 1 else ("", similarity)
        return GalleryMatch(
            animal_id=animal_id,
            similarity=similarity,
            separation=similarity - second,
            runner_up=runner_up,
        )
