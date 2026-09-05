"""Pass segmentation.

Entry and exit boundaries are configuration, not an assumption about geometry,
because the public sources this change targets do not share a lane. A pass that
lacks either crossing is labelled partial and its boundaries are left missing —
extrapolating to a crossing that was never observed would manufacture the exact
quantity the feature set is trying to measure.
"""

from __future__ import annotations

from ..config import ResolvedConfig
from ..perception.schemas import Tracklet
from .schemas import LanePass, PassCompleteness

__all__ = ["segment_passes"]


def _axis_positions(tracklet: Tracklet, axis: str, width: int, height: int) -> list[float]:
    extent = float(width if axis == "x" else height) or 1.0
    positions = []
    for detection in tracklet.detections:
        cx, cy = detection.box.centre
        positions.append((cx if axis == "x" else cy) / extent)
    return positions


def segment_passes(
    tracklet: Tracklet,
    config: ResolvedConfig,
    *,
    frame_width: int,
    frame_height: int,
) -> list[LanePass]:
    """Split one tracklet into lane passes.

    A complete pass spans exactly the entry crossing and the exit crossing.
    Anything else is emitted partial, naming which boundary is missing.
    """
    phenotype = config.phenotype
    positions = _axis_positions(tracklet, phenotype.boundary_axis, frame_width, frame_height)
    if not positions:
        return []

    # Work in a coordinate that always increases from entry towards exit, so the
    # same logic serves a lane travelled in either direction.
    sign = 1.0 if phenotype.exit_boundary >= phenotype.entry_boundary else -1.0
    progress = [sign * p for p in positions]
    entry_level = sign * phenotype.entry_boundary
    exit_level = sign * phenotype.exit_boundary

    indices = tracklet.frame_indices
    passes: list[LanePass] = []
    cursor = 0
    sequence = 0

    while cursor < len(progress):
        entry_at = _first_crossing(progress, entry_level, start=cursor)
        exit_at = (
            _first_crossing(progress, exit_level, start=entry_at)
            if entry_at is not None
            else _first_crossing(progress, exit_level, start=cursor)
        )

        if entry_at is not None and exit_at is not None:
            passes.append(
                _build(
                    tracklet,
                    sequence,
                    start=entry_at,
                    end=exit_at,
                    completeness=PassCompleteness.COMPLETE,
                    entry_index=indices[entry_at],
                    exit_index=indices[exit_at],
                    direction="forward" if sign > 0 else "reverse",
                )
            )
            sequence += 1
            cursor = exit_at + 1
            continue

        missing = "entry" if entry_at is None else "exit"
        if entry_at is None and exit_at is None:
            missing = "entry and exit"
        passes.append(
            _build(
                tracklet,
                sequence,
                start=cursor,
                end=len(progress) - 1,
                completeness=PassCompleteness.PARTIAL,
                entry_index=indices[entry_at] if entry_at is not None else None,
                exit_index=indices[exit_at] if exit_at is not None else None,
                direction="forward" if sign > 0 else "reverse",
                missing_boundary=missing,
            )
        )
        break

    return passes


def _first_crossing(progress: list[float], level: float, *, start: int) -> int | None:
    """Index at which the track crosses ``level`` upward, at or after ``start``.

    A track already past the level at ``start`` has not been seen to cross it,
    and returns None rather than a fabricated crossing.
    """
    for index in range(max(start, 1), len(progress)):
        if progress[index - 1] < level <= progress[index]:
            return index
    return None


def _build(
    tracklet: Tracklet,
    sequence: int,
    *,
    start: int,
    end: int,
    completeness: PassCompleteness,
    entry_index: int | None,
    exit_index: int | None,
    direction: str,
    missing_boundary: str = "",
) -> LanePass:
    detections = tracklet.detections[start : end + 1]
    first = detections[0]
    last = detections[-1]
    reliable = tracklet.timestamps_reliable
    return LanePass(
        pass_id=f"{tracklet.tracklet_id}:p{sequence:03d}",
        tracklet_id=tracklet.tracklet_id,
        source_id=tracklet.source_id,
        camera_id=tracklet.camera_id,
        site_key=tracklet.site_key,
        animal_set_key=tracklet.animal_set_key,
        day_key=tracklet.day_key,
        first_frame_index=first.frame_index,
        last_frame_index=last.frame_index,
        completeness=completeness,
        entry_frame_index=entry_index,
        exit_frame_index=exit_index,
        first_timestamp=first.provenance.capture_timestamp if reliable else None,
        last_timestamp=last.provenance.capture_timestamp if reliable else None,
        frame_count=len(detections),
        direction=direction,
        missing_boundary=missing_boundary,
    )
