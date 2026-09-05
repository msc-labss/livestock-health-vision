"""Identity anchoring: anchor path, fallback, unresolved, provenance, conflicts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from lhv.identity import (
    AnchorRecord,
    AssignmentMethod,
    ColourHistogramEmbedding,
    DatasetLabelAnchorSource,
    IdentityAssignment,
    IdentityResolver,
    InMemoryAnchorSource,
    ReferenceGallery,
    UnresolvedReason,
)
from lhv.ingest import FrameProvenance
from lhv.perception import BoundingBox, Detection, TerminationReason, Tracklet

BASE = datetime(2024, 3, 1, 6, 30, tzinfo=UTC)
NOW = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def _tracklet(
    tracklet_id: str = "s:t00001",
    *,
    start: int = 0,
    end: int = 5,
    site_key: str = "site-a",
    camera_id: str = "cam-1",
) -> Tracklet:
    detections = []
    for index in range(start, end + 1):
        provenance = FrameProvenance(
            source_id="s",
            camera_id=camera_id,
            frame_index=index,
            site_key=site_key,
            animal_set_key="herd-a",
            day_key="2024-03-01",
            capture_timestamp=BASE + timedelta(seconds=index),
        )
        detections.append(
            Detection(
                detection_id=f"s#{index}:{tracklet_id}",
                provenance=provenance,
                box=BoundingBox(10, 10 + index, 60, 70 + index),
                label="animal",
                confidence=0.9,
                model_identity="stub@1",
            )
        )
    return Tracklet(
        tracklet_id=tracklet_id,
        source_id="s",
        camera_id=camera_id,
        site_key=site_key,
        animal_set_key="herd-a",
        day_key="2024-03-01",
        first_frame_index=start,
        last_frame_index=end,
        detections=tuple(detections),
        termination_reason=TerminationReason.EXIT,
        model_identity="stub@1",
        first_timestamp=BASE + timedelta(seconds=start),
        last_timestamp=BASE + timedelta(seconds=end),
    )


def _anchor(
    animal_id: str, *, start: int = 0, end: int = 5, source: str = "parlour"
) -> AnchorRecord:
    return AnchorRecord(
        animal_id=animal_id,
        anchor_source=source,
        site_key="site-a",
        observed_from=BASE + timedelta(seconds=start),
        observed_to=BASE + timedelta(seconds=end),
        camera_id="cam-1",
        reader_id="reader-1",
    )


# -- 5.1 external anchor is the primary path --------------------------------


def test_external_identifier_resolves_the_tracklet(config) -> None:
    resolver = IdentityResolver(
        config, anchor_source=InMemoryAnchorSource([_anchor("cow-17")]), now=NOW
    )
    assignment = resolver.resolve(_tracklet())
    assert assignment.method is AssignmentMethod.EXTERNAL_ANCHOR
    assert assignment.animal_id == "cow-17"
    assert assignment.anchor_source == "parlour"
    assert resolver.report.anchored == 1


def test_dataset_labels_are_routed_through_the_anchor_interface(config) -> None:
    """P0 substitutes ground truth for a farm identifier without changing the flow."""
    tracklets = [_tracklet("s:t1"), _tracklet("s:t2", start=20, end=25)]
    source = DatasetLabelAnchorSource.from_labelled_tracklets(
        {"s:t1": "animal-a", "s:t2": "animal-b"}, tracklets, dataset_name="cattleeyeview"
    )
    resolver = IdentityResolver(config, anchor_source=source, now=NOW)
    assignments = resolver.resolve_all(tracklets)

    assert [a.animal_id for a in assignments] == ["animal-a", "animal-b"]
    for assignment in assignments:
        assert assignment.method is AssignmentMethod.EXTERNAL_ANCHOR
        # The anchor source names itself as a dataset label, so nothing can
        # later mistake it for a farm identifier stream.
        assert assignment.anchor_source == "dataset-label:cattleeyeview"


def test_an_anchor_at_another_site_does_not_resolve_a_tracklet(config) -> None:
    resolver = IdentityResolver(
        config, anchor_source=InMemoryAnchorSource([_anchor("cow-17")]), now=NOW
    )
    assignment = resolver.resolve(_tracklet(site_key="site-b"))
    assert assignment.method is AssignmentMethod.UNRESOLVED


# -- 5.2 ambiguous match is not resolved by guessing ------------------------


def test_ambiguous_anchor_leaves_the_tracklet_unresolved_with_candidates(config) -> None:
    resolver = IdentityResolver(
        config,
        anchor_source=InMemoryAnchorSource([_anchor("cow-17"), _anchor("cow-18")]),
        now=NOW,
    )
    assignment = resolver.resolve(_tracklet())
    assert assignment.method is AssignmentMethod.UNRESOLVED
    assert assignment.unresolved_reason is UnresolvedReason.AMBIGUOUS_ANCHOR
    assert set(assignment.candidate_animal_ids) == {"cow-17", "cow-18"}
    assert assignment.animal_id == ""


def test_an_unresolved_assignment_may_not_carry_an_identity() -> None:
    with pytest.raises(ValueError, match="must not carry a provisional identity"):
        IdentityAssignment(
            tracklet_id="t",
            method=AssignmentMethod.UNRESOLVED,
            assigned_at=NOW,
            site_key="site-a",
            day_key="2024-03-01",
            animal_id="cow-17",
        )


# -- 5.3 visual fallback is marked as fallback ------------------------------


def _patch(colour: tuple[int, int, int]) -> np.ndarray:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :] = colour
    return image


def test_visual_fallback_is_marked_and_carries_a_confidence(config) -> None:
    backend = ColourHistogramEmbedding()
    gallery = ReferenceGallery(backend=backend)
    gallery.enrol_image("cow-21", _patch((10, 200, 30)))
    gallery.enrol_image("cow-22", _patch((200, 10, 30)))

    resolver = IdentityResolver(config, anchor_source=None, gallery=gallery, now=NOW)
    assignment = resolver.resolve(_tracklet(), crop=_patch((10, 200, 30)))

    assert assignment.method is AssignmentMethod.VISUAL_FALLBACK
    assert assignment.animal_id == "cow-21"
    assert 0.0 < assignment.confidence <= 1.0
    assert resolver.report.fallback == 1


def test_consumers_can_filter_on_the_assignment_method(config) -> None:
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-21", _patch((10, 200, 30)))
    resolver = IdentityResolver(
        config,
        anchor_source=InMemoryAnchorSource([_anchor("cow-17", start=100, end=105)]),
        gallery=gallery,
        now=NOW,
    )
    assignments = resolver.resolve_all([_tracklet("s:t1")], crops={"s:t1": _patch((10, 200, 30))})
    fallback_only = [a for a in assignments if a.method is AssignmentMethod.VISUAL_FALLBACK]
    assert len(fallback_only) == 1


def test_the_anchor_is_preferred_over_the_visual_path(config) -> None:
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-99", _patch((10, 200, 30)))
    resolver = IdentityResolver(
        config,
        anchor_source=InMemoryAnchorSource([_anchor("cow-17")]),
        gallery=gallery,
        now=NOW,
    )
    assignment = resolver.resolve(_tracklet(), crop=_patch((10, 200, 30)))
    assert assignment.method is AssignmentMethod.EXTERNAL_ANCHOR
    assert assignment.animal_id == "cow-17"


# -- 5.4 unresolved is a valid outcome --------------------------------------


def test_no_anchor_and_no_gallery_yields_unresolved(config) -> None:
    resolver = IdentityResolver(config, now=NOW)
    assignment = resolver.resolve(_tracklet())
    assert assignment.method is AssignmentMethod.UNRESOLVED
    assert assignment.unresolved_reason is UnresolvedReason.NO_CANDIDATE
    assert not assignment.enters_time_series


def test_a_match_below_the_confidence_floor_is_unresolved(config) -> None:
    import dataclasses

    strict = dataclasses.replace(
        config,
        identity=dataclasses.replace(
            config.identity, confidence_floor=0.999, reid_similarity_floor=0.999
        ),
    )
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-21", _patch((10, 200, 30)))

    resolver = IdentityResolver(strict, gallery=gallery, now=NOW)
    assignment = resolver.resolve(_tracklet(), crop=_patch((200, 10, 30)))

    assert assignment.method is AssignmentMethod.UNRESOLVED
    assert assignment.unresolved_reason is UnresolvedReason.BELOW_CONFIDENCE_FLOOR
    assert not assignment.enters_time_series
    # Excluded from the series, but the near miss is still on the record.
    assert assignment.candidate_animal_ids == ("cow-21",)
    assert assignment.confidence > 0.0


def test_unresolved_assignments_remain_retrievable_for_audit(config) -> None:
    resolver = IdentityResolver(config, now=NOW)
    assignment = resolver.resolve(_tracklet())
    restored = IdentityAssignment.from_dict(assignment.to_dict())
    assert restored == assignment
    assert restored.tracklet_id == "s:t00001"


# -- 5.5 assignment provenance ----------------------------------------------


def test_assignment_provenance_is_retrievable_without_reprocessing_video(config) -> None:
    resolver = IdentityResolver(
        config, anchor_source=InMemoryAnchorSource([_anchor("cow-17")]), now=NOW
    )
    encoded = resolver.resolve(_tracklet()).to_dict()

    assert encoded["method"] == "external_anchor"
    assert encoded["confidence"] == 1.0
    assert encoded["assigned_at"] == NOW.isoformat()
    assert encoded["evidence_reference"].startswith("anchor:parlour:reader-1:")
    assert encoded["schema_version"] == IdentityAssignment.SCHEMA_VERSION


def test_visual_assignment_provenance_names_the_embedding_model(config) -> None:
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding(version="7"))
    gallery.enrol_image("cow-21", _patch((10, 200, 30)))
    resolver = IdentityResolver(config, gallery=gallery, now=NOW)
    assignment = resolver.resolve(_tracklet(), crop=_patch((10, 200, 30)))
    assert assignment.evidence_reference == "visual:colour-histogram@7"


# -- 5.6 identity conflict is surfaced --------------------------------------


def test_two_overlapping_tracklets_resolving_to_one_animal_are_both_withheld(config) -> None:
    left = _tracklet("s:t1", start=0, end=10)
    right = _tracklet("s:t2", start=5, end=15, camera_id="cam-1")
    anchors = InMemoryAnchorSource(
        [_anchor("cow-17", start=0, end=10), _anchor("cow-17", start=5, end=15)]
    )
    resolver = IdentityResolver(config, anchor_source=anchors, now=NOW)
    assignments = resolver.resolve_all([left, right])

    assert all(a.method is AssignmentMethod.UNRESOLVED for a in assignments)
    assert all(a.unresolved_reason is UnresolvedReason.CONFLICT for a in assignments)
    assert all(not a.enters_time_series for a in assignments)

    assert len(resolver.conflicts) == 1
    conflict = resolver.conflicts[0]
    assert conflict.animal_id == "cow-17"
    assert set(conflict.tracklet_ids) == {"s:t1", "s:t2"}
    assert resolver.report.conflicts == 1
    assert resolver.report.anchored == 0


def test_tracklets_that_do_not_overlap_in_time_are_not_a_conflict(config) -> None:
    first = _tracklet("s:t1", start=0, end=5)
    second = _tracklet("s:t2", start=20, end=25)
    anchors = InMemoryAnchorSource(
        [_anchor("cow-17", start=0, end=5), _anchor("cow-17", start=20, end=25)]
    )
    resolver = IdentityResolver(config, anchor_source=anchors, now=NOW)
    assignments = resolver.resolve_all([first, second])

    assert all(a.method is AssignmentMethod.EXTERNAL_ANCHOR for a in assignments)
    assert resolver.conflicts == []
    assert resolver.report.anchored == 2


def test_the_conflict_record_names_both_tracklets_and_is_serialisable(config) -> None:
    from lhv.identity import IdentityConflict

    left = _tracklet("s:t1", start=0, end=10)
    right = _tracklet("s:t2", start=5, end=15)
    anchors = InMemoryAnchorSource(
        [_anchor("cow-17", start=0, end=10), _anchor("cow-17", start=5, end=15)]
    )
    resolver = IdentityResolver(config, anchor_source=anchors, now=NOW)
    resolver.resolve_all([left, right])

    conflict = resolver.conflicts[0]
    assert IdentityConflict.from_dict(conflict.to_dict()) == conflict
    assert conflict.to_dict()["tracklet_ids"] == ["s:t1", "s:t2"]


# -- what the fallback's confidence is worth --------------------------------


def test_the_gallery_reports_its_margin_over_the_runner_up() -> None:
    from lhv.identity import GalleryMatch

    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-a", _patch((10, 200, 30)))
    gallery.enrol_image("cow-b", _patch((200, 10, 30)))

    match = gallery.best_match(ColourHistogramEmbedding().embed(_patch((10, 200, 30))))
    assert isinstance(match, GalleryMatch)
    assert match.animal_id == "cow-a"
    assert match.runner_up == "cow-b"
    assert match.separation > 0.0
    assert match.separation == pytest.approx(
        match.similarity - _second(gallery, _patch((10, 200, 30)))
    )


def _second(gallery, image):
    ranked = gallery.rank(ColourHistogramEmbedding().embed(image))
    return ranked[1][1]


def test_a_single_enrolled_animal_has_no_margin() -> None:
    """Nothing to beat means nothing has been shown."""
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-a", _patch((10, 200, 30)))
    match = gallery.best_match(ColourHistogramEmbedding().embed(_patch((10, 200, 30))))
    assert match.separation == 0.0
    assert match.runner_up == ""


def test_the_assignment_records_the_separation(config) -> None:
    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    gallery.enrol_image("cow-a", _patch((10, 200, 30)))
    gallery.enrol_image("cow-b", _patch((200, 10, 30)))

    resolver = IdentityResolver(config, gallery=gallery, now=NOW)
    assignment = resolver.resolve(_tracklet(), crop=_patch((10, 200, 30)))
    assert assignment.method is AssignmentMethod.VISUAL_FALLBACK
    assert assignment.separation > 0.0
    assert IdentityAssignment.from_dict(assignment.to_dict()).separation == pytest.approx(
        assignment.separation
    )


def test_a_margin_floor_can_refuse_an_indistinct_match(config) -> None:
    """On a saturated embedding a similarity floor filters nothing; this does."""
    import dataclasses

    gallery = ReferenceGallery(backend=ColourHistogramEmbedding())
    # Two nearly identical references: whichever wins, it barely wins.
    gallery.enrol_image("cow-a", _patch((100, 100, 100)))
    gallery.enrol_image("cow-b", _patch((101, 100, 100)))

    permissive = IdentityResolver(config, gallery=gallery, now=NOW).resolve(
        _tracklet(), crop=_patch((100, 100, 100))
    )
    assert permissive.method is AssignmentMethod.VISUAL_FALLBACK

    strict = dataclasses.replace(
        config, identity=dataclasses.replace(config.identity, reid_margin_floor=0.5)
    )
    refused = IdentityResolver(strict, gallery=gallery, now=NOW).resolve(
        _tracklet(), crop=_patch((100, 100, 100))
    )
    assert refused.method is AssignmentMethod.UNRESOLVED
    assert refused.unresolved_reason is UnresolvedReason.BELOW_CONFIDENCE_FLOOR
    # The near miss is still on the record.
    assert refused.candidate_animal_ids
    assert strict.digest != config.digest, "the floor must be visible in the digest"


def test_the_margin_floor_defaults_to_gating_nothing(config) -> None:
    """The right value depends on the embedding, so the default assumes none."""
    assert config.identity.reid_margin_floor == 0.0
