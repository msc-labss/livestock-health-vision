"""Stage boundary: tracklets -> animal identity assignments.

External anchor first, visual re-identification as a marked fallback, and
unresolved as a real outcome rather than a guess.
"""

from .anchor import AnchorSource, DatasetLabelAnchorSource, InMemoryAnchorSource
from .reid import (
    ColourHistogramEmbedding,
    GalleryMatch,
    ReferenceGallery,
    cosine_similarity,
)
from .resolve import IdentityReport, IdentityResolver
from .schemas import (
    AnchorRecord,
    AssignmentMethod,
    IdentityAssignment,
    IdentityConflict,
    UnresolvedReason,
)

__all__ = [
    "AnchorRecord",
    "AnchorSource",
    "AssignmentMethod",
    "ColourHistogramEmbedding",
    "DatasetLabelAnchorSource",
    "GalleryMatch",
    "IdentityAssignment",
    "IdentityConflict",
    "IdentityReport",
    "IdentityResolver",
    "InMemoryAnchorSource",
    "ReferenceGallery",
    "UnresolvedReason",
    "cosine_similarity",
]
