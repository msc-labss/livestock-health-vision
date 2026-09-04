"""Resolving a tracklet to a stable animal identity.

Order of preference is fixed: external anchor, then visual fallback, then
unresolved. Unresolved is a real outcome and never a provisional identity,
because a wrong identity contaminates a longitudinal baseline in a way a missing
one does not.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from ..config import ResolvedConfig
from ..perception.schemas import Tracklet
from .anchor import AnchorSource
from .reid import ReferenceGallery
from .schemas import (
    AssignmentMethod,
    IdentityAssignment,
    IdentityConflict,
    UnresolvedReason,
)

__all__ = ["IdentityReport", "IdentityResolver"]


@dataclass
class IdentityReport:
    anchored: int = 0
    fallback: int = 0
    unresolved: int = 0
    conflicts: int = 0
    unresolved_reasons: dict[str, int] = field(default_factory=dict)

    def note_unresolved(self, reason: UnresolvedReason) -> None:
        self.unresolved += 1
        self.unresolved_reasons[reason.value] = self.unresolved_reasons.get(reason.value, 0) + 1

    def describe(self) -> str:
        reasons = ", ".join(f"{k}={v}" for k, v in sorted(self.unresolved_reasons.items()))
        return (
            f"identity: {self.anchored} anchored, {self.fallback} visual fallback, "
            f"{self.unresolved} unresolved [{reasons}], {self.conflicts} conflict(s)"
        )


class IdentityResolver:
    """Resolves tracklets, then withholds any identity claimed twice at once."""

    def __init__(
        self,
        config: ResolvedConfig,
        *,
        anchor_source: AnchorSource | None = None,
        gallery: ReferenceGallery | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.anchor_source = anchor_source
        self.gallery = gallery
        self._now = now
        self.report = IdentityReport()
        self.conflicts: list[IdentityConflict] = []

    def _timestamp(self) -> datetime:
        return self._now or datetime.now(UTC)

    # -- single tracklet ----------------------------------------------------

    def resolve(
        self,
        tracklet: Tracklet,
        *,
        crop: np.ndarray | None = None,
    ) -> IdentityAssignment:
        anchored = self._try_anchor(tracklet)
        if anchored is not None:
            return anchored
        fallback = self._try_visual(tracklet, crop)
        if fallback is not None:
            return fallback
        return self._unresolved(tracklet, UnresolvedReason.NO_CANDIDATE)

    def _try_anchor(self, tracklet: Tracklet) -> IdentityAssignment | None:
        if self.anchor_source is None:
            return None
        if tracklet.first_timestamp is None or tracklet.last_timestamp is None:
            return None

        candidates = self.anchor_source.records_for(
            site_key=tracklet.site_key,
            start=tracklet.first_timestamp,
            end=tracklet.last_timestamp,
            camera_id=tracklet.camera_id,
        )
        distinct = sorted({record.animal_id for record in candidates})
        if not distinct:
            return None
        if len(distinct) > 1:
            # More than one identifier fits the window. Guessing here is how a
            # baseline silently becomes two animals' data averaged together.
            self.report.note_unresolved(UnresolvedReason.AMBIGUOUS_ANCHOR)
            return IdentityAssignment(
                tracklet_id=tracklet.tracklet_id,
                method=AssignmentMethod.UNRESOLVED,
                assigned_at=self._timestamp(),
                site_key=tracklet.site_key,
                day_key=tracklet.day_key,
                unresolved_reason=UnresolvedReason.AMBIGUOUS_ANCHOR,
                candidate_animal_ids=tuple(distinct),
                anchor_source=self.anchor_source.source_name,
                evidence_reference=f"anchor:{self.anchor_source.source_name}",
            )

        record = next(r for r in candidates if r.animal_id == distinct[0])
        self.report.anchored += 1
        return IdentityAssignment(
            tracklet_id=tracklet.tracklet_id,
            method=AssignmentMethod.EXTERNAL_ANCHOR,
            assigned_at=self._timestamp(),
            site_key=tracklet.site_key,
            day_key=tracklet.day_key,
            animal_id=record.animal_id,
            confidence=record.confidence,
            anchor_source=record.anchor_source,
            evidence_reference=(
                f"anchor:{record.anchor_source}:{record.reader_id or record.camera_id}:"
                f"{record.observed_from.isoformat()}"
            ),
        )

    def _try_visual(self, tracklet: Tracklet, crop: np.ndarray | None) -> IdentityAssignment | None:
        if self.gallery is None or crop is None or len(self.gallery) == 0:
            return None

        embedding = self.gallery.backend.embed(crop)
        match = self.gallery.best_match(embedding)
        if match is None:
            return None
        animal_id, similarity = match

        floor = max(
            self.config.identity.confidence_floor, self.config.identity.reid_similarity_floor
        )
        if similarity < floor:
            self.report.note_unresolved(UnresolvedReason.BELOW_CONFIDENCE_FLOOR)
            return IdentityAssignment(
                tracklet_id=tracklet.tracklet_id,
                method=AssignmentMethod.UNRESOLVED,
                assigned_at=self._timestamp(),
                site_key=tracklet.site_key,
                day_key=tracklet.day_key,
                unresolved_reason=UnresolvedReason.BELOW_CONFIDENCE_FLOOR,
                candidate_animal_ids=(animal_id,),
                evidence_reference=f"visual:{self.gallery.backend.model_identity}",
                confidence=similarity,
            )

        self.report.fallback += 1
        return IdentityAssignment(
            tracklet_id=tracklet.tracklet_id,
            method=AssignmentMethod.VISUAL_FALLBACK,
            assigned_at=self._timestamp(),
            site_key=tracklet.site_key,
            day_key=tracklet.day_key,
            animal_id=animal_id,
            confidence=similarity,
            evidence_reference=f"visual:{self.gallery.backend.model_identity}",
        )

    def _unresolved(self, tracklet: Tracklet, reason: UnresolvedReason) -> IdentityAssignment:
        self.report.note_unresolved(reason)
        return IdentityAssignment(
            tracklet_id=tracklet.tracklet_id,
            method=AssignmentMethod.UNRESOLVED,
            assigned_at=self._timestamp(),
            site_key=tracklet.site_key,
            day_key=tracklet.day_key,
            unresolved_reason=reason,
        )

    # -- whole source -------------------------------------------------------

    def resolve_all(
        self,
        tracklets: Iterable[Tracklet],
        *,
        crops: dict[str, np.ndarray] | None = None,
    ) -> list[IdentityAssignment]:
        """Resolve every tracklet, then withhold identities claimed twice at once."""
        crops = crops or {}
        tracklets = list(tracklets)
        assignments = [
            self.resolve(tracklet, crop=crops.get(tracklet.tracklet_id)) for tracklet in tracklets
        ]
        return self._withhold_conflicts(tracklets, assignments)

    def _withhold_conflicts(
        self, tracklets: list[Tracklet], assignments: list[IdentityAssignment]
    ) -> list[IdentityAssignment]:
        by_id = {t.tracklet_id: t for t in tracklets}
        by_animal: dict[str, list[IdentityAssignment]] = {}
        for assignment in assignments:
            if assignment.resolved:
                by_animal.setdefault(assignment.animal_id, []).append(assignment)

        conflicted: set[str] = set()
        for animal_id, claims in sorted(by_animal.items()):
            for i in range(len(claims)):
                for j in range(i + 1, len(claims)):
                    left = by_id[claims[i].tracklet_id]
                    right = by_id[claims[j].tracklet_id]
                    if not left.overlaps_in_time(right):
                        continue
                    pair = (left.tracklet_id, right.tracklet_id)
                    conflicted.update(pair)
                    self.conflicts.append(
                        IdentityConflict(
                            animal_id=animal_id,
                            tracklet_ids=tuple(sorted(pair)),
                            detected_at=self._timestamp(),
                            site_key=left.site_key,
                            day_key=left.day_key,
                            note=(
                                "two tracklets overlapping in time resolved to one animal; both "
                                "are withheld from the time series until resolved"
                            ),
                        )
                    )
                    self.report.conflicts += 1

        if not conflicted:
            return assignments

        withheld: list[IdentityAssignment] = []
        for assignment in assignments:
            if assignment.tracklet_id not in conflicted:
                withheld.append(assignment)
                continue
            if assignment.method is AssignmentMethod.EXTERNAL_ANCHOR:
                self.report.anchored -= 1
            elif assignment.method is AssignmentMethod.VISUAL_FALLBACK:
                self.report.fallback -= 1
            self.report.note_unresolved(UnresolvedReason.CONFLICT)
            withheld.append(
                IdentityAssignment(
                    tracklet_id=assignment.tracklet_id,
                    method=AssignmentMethod.UNRESOLVED,
                    assigned_at=assignment.assigned_at,
                    site_key=assignment.site_key,
                    day_key=assignment.day_key,
                    confidence=assignment.confidence,
                    anchor_source=assignment.anchor_source,
                    evidence_reference=assignment.evidence_reference,
                    unresolved_reason=UnresolvedReason.CONFLICT,
                    candidate_animal_ids=(assignment.animal_id,),
                )
            )
        return withheld
