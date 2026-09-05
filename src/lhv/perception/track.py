"""Motion-based tracking into tracklets.

Association uses motion alone — overlap against a constant-velocity prediction —
with no appearance embedding. The target geometry is a lane carrying one animal
at a time in a known direction, so appearance association solves a problem the
geometry has already removed, at a compute cost the edge phase will not have.
Appearance is confined to the identity fallback.

A tracklet records why it ended. Exit, occlusion loss and detection failure are
different events with different remedies, and collapsing them into "lost" throws
away the only diagnostic the stage produces.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..config import ResolvedConfig
from .schemas import BoundingBox, Detection, FrameDetections, TerminationReason, Tracklet

__all__ = ["Tracker", "TrackerReport"]


@dataclass
class _ActiveTrack:
    track_index: int
    detections: list[Detection] = field(default_factory=list)
    last_frame_index: int = -1
    velocity: tuple[float, float] = (0.0, 0.0)
    missed: int = 0
    # Set when another detection was found overlapping the predicted position
    # while this track was unmatched — the signature of an occlusion rather than
    # a detector miss.
    occluded_while_missing: bool = False

    @property
    def last_box(self) -> BoundingBox:
        return self.detections[-1].box

    def predicted_box(self, frames_ahead: int = 1) -> BoundingBox:
        dx, dy = self.velocity
        return self.last_box.translated(dx * frames_ahead, dy * frames_ahead)

    def extend(self, detection: Detection) -> None:
        if self.detections:
            previous = self.detections[-1]
            gap = max(detection.frame_index - previous.frame_index, 1)
            px, py = previous.box.centre
            cx, cy = detection.box.centre
            self.velocity = ((cx - px) / gap, (cy - py) / gap)
        self.detections.append(detection)
        self.last_frame_index = detection.frame_index
        self.missed = 0
        self.occluded_while_missing = False


@dataclass
class TrackerReport:
    tracklets_formed: int = 0
    terminations: dict[str, int] = field(default_factory=dict)

    def record(self, reason: TerminationReason) -> None:
        self.tracklets_formed += 1
        self.terminations[reason.value] = self.terminations.get(reason.value, 0) + 1

    def describe(self) -> str:
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.terminations.items()))
        return f"{self.tracklets_formed} tracklet(s) [{parts}]"


class Tracker:
    """Groups per-frame detections into tracklets within one source."""

    def __init__(self, config: ResolvedConfig, *, source_id: str = "") -> None:
        self.config = config
        self.source_id = source_id
        self.report = TrackerReport()

    def track(self, frames: list[FrameDetections]) -> list[Tracklet]:
        """Form tracklets from an ordered sequence of per-frame detection records."""
        perception = self.config.perception
        active: list[_ActiveTrack] = []
        finished: list[Tracklet] = []
        next_index = 0
        frame_width = frame_height = 0

        for record in frames:
            frame_width = record.frame_width or frame_width
            frame_height = record.frame_height or frame_height
            unmatched = list(record.detections)

            # Greedy association by overlap with the constant-velocity prediction.
            # Ordered by descending overlap so the strongest pairing wins first.
            pairs = []
            for track in active:
                predicted = track.predicted_box(max(track.missed, 1))
                for detection in unmatched:
                    score = predicted.iou(detection.box)
                    if score >= perception.min_track_iou:
                        pairs.append((score, track, detection))
            pairs.sort(key=lambda item: (-item[0], item[1].track_index))

            claimed_tracks: set[int] = set()
            claimed_detections: set[str] = set()
            for _, track, detection in pairs:
                if track.track_index in claimed_tracks:
                    continue
                if detection.detection_id in claimed_detections:
                    continue
                track.extend(detection)
                claimed_tracks.add(track.track_index)
                claimed_detections.add(detection.detection_id)

            for track in active:
                if track.track_index in claimed_tracks:
                    continue
                track.missed += 1
                # Something else sitting where this track should be is what an
                # occlusion looks like from a motion tracker's point of view.
                predicted = track.predicted_box(track.missed)
                for detection in record.detections:
                    if predicted.iou(detection.box) > 0.0:
                        track.occluded_while_missing = True

            still_active = []
            for track in active:
                if track.missed > perception.max_track_gap_frames:
                    finished.append(
                        self._finish(track, frame_width, frame_height, at_source_end=False)
                    )
                else:
                    still_active.append(track)
            active = still_active

            for detection in record.detections:
                if detection.detection_id in claimed_detections:
                    continue
                track = _ActiveTrack(track_index=next_index)
                next_index += 1
                track.extend(detection)
                active.append(track)

        for track in active:
            finished.append(self._finish(track, frame_width, frame_height, at_source_end=True))

        finished.sort(key=lambda t: (t.first_frame_index, t.tracklet_id))
        return finished

    # -- termination --------------------------------------------------------

    def _termination_reason(
        self,
        track: _ActiveTrack,
        frame_width: int,
        frame_height: int,
        *,
        at_source_end: bool,
    ) -> TerminationReason:
        if at_source_end:
            return TerminationReason.SOURCE_END
        if (
            frame_width
            and frame_height
            and track.last_box.touches_border(frame_width, frame_height)
        ):
            return TerminationReason.EXIT
        # Something else standing where this track was predicted to be is what
        # an occlusion looks like to a motion tracker. Nothing there at all is a
        # detector miss. The two call for different remedies, so they are not
        # collapsed into one "lost".
        if track.occluded_while_missing:
            return TerminationReason.OCCLUSION_LOSS
        return TerminationReason.DETECTION_FAILURE

    def _finish(
        self,
        track: _ActiveTrack,
        frame_width: int,
        frame_height: int,
        *,
        at_source_end: bool,
    ) -> Tracklet:
        reason = self._termination_reason(
            track, frame_width, frame_height, at_source_end=at_source_end
        )
        first = track.detections[0]
        last = track.detections[-1]
        provenance = first.provenance
        source_id = self.source_id or provenance.source_id
        self.report.record(reason)
        reliable = all(d.provenance.timestamp_reliable for d in track.detections)
        return Tracklet(
            tracklet_id=f"{source_id}:t{track.track_index:05d}",
            source_id=source_id,
            camera_id=provenance.camera_id,
            site_key=provenance.site_key,
            animal_set_key=provenance.animal_set_key,
            day_key=provenance.day_key,
            first_frame_index=first.frame_index,
            last_frame_index=last.frame_index,
            detections=tuple(track.detections),
            termination_reason=reason,
            model_identity=first.model_identity,
            first_timestamp=first.provenance.capture_timestamp if reliable else None,
            last_timestamp=last.provenance.capture_timestamp if reliable else None,
            timestamps_reliable=reliable,
        )
