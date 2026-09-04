"""Reading COCO-format keypoint annotations that carry track identifiers.

COCO is a standard, and this reads the standard. One deviation is handled
explicitly rather than assumed away: in the CattleEyeView release the
``image_id`` on an annotation does not reference ``images[].id``. It disagrees
with the annotation's own ``file_name`` in 24,051 of 24,054 cases, because it
is a running index over the annotation list. Joining on ``image_id``, which is
what the format invites, pairs every annotation with the wrong frame and
produces an evaluation that looks fine and measures nothing.

So the join is on ``file_name`` by default, and asking for the ``image_id``
join makes the caller say so.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from ..errors import RegistrationError
from ..evaluation.metrics import LabelledBox, LabelledKeypoint
from ..perception.schemas import BoundingBox
from ..profiles import SkeletonDefinition

__all__ = ["CocoKeypointLabels", "LabelledPoseBackend", "load_coco_keypoints"]

_FRAME = re.compile(r"^(?P<sequence>[^/]+?)(?:\.mp4)?/(?P<frame>\d+)\.(?:jpg|jpeg|png)$")

# The release's per-keypoint visibility flag.
_NOT_LABELLED = 0.0
_OCCLUDED = 1.0  # labelled, but the annotator could not see it
_VISIBLE = 2.0


@dataclass(frozen=True)
class CocoKeypointLabels:
    """Ground truth for one split, keyed the way this pipeline addresses frames."""

    boxes: tuple[LabelledBox, ...]
    keypoints: tuple[LabelledKeypoint, ...]
    split: str = ""
    sequences: tuple[str, ...] = ()

    def detector_boxes(self) -> dict[tuple[str, int], list[tuple[BoundingBox, str]]]:
        """Boxes in the shape an annotation-backed detector serves."""
        served: dict[tuple[str, int], list[tuple[BoundingBox, str]]] = {}
        for label in self.boxes:
            served.setdefault((label.source_id, label.frame_index), []).append(
                (label.box, "animal")
            )
        return served

    def identity_by_track(self) -> dict[str, str]:
        """Track identifier -> the animal identity it stands for.

        In this release the track identifier is the identity: an instance is one
        animal's appearance in one sequence. It is not an animal followed across
        sequences, and nothing here pretends otherwise.
        """
        return {label.track_id: label.animal_id for label in self.boxes if label.track_id}

    def for_sequences(self, sequences) -> CocoKeypointLabels:
        keep = set(sequences)
        return CocoKeypointLabels(
            boxes=tuple(b for b in self.boxes if b.source_id.rsplit("/", 1)[-1] in keep),
            keypoints=tuple(k for k in self.keypoints if k.source_id.rsplit("/", 1)[-1] in keep),
            split=self.split,
            sequences=tuple(sorted(keep)),
        )


def load_coco_keypoints(
    path: str | Path,
    *,
    dataset_name: str,
    skeleton: SkeletonDefinition,
    frame_offset: int = -1,
    join_on: str = "file_name",
) -> CocoKeypointLabels:
    """Read one COCO keypoint file into labels addressed by source and frame.

    ``frame_offset`` converts the release's frame numbering to this pipeline's.
    CattleEyeView numbers its stills from one against a video indexed from
    zero, so the default is -1; the release's own ``frame_id`` agrees with that
    for all 30,703 of its images.
    """
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))

    if join_on not in {"file_name", "image_id"}:
        raise RegistrationError(f"unknown COCO join key {join_on!r}")

    by_image_id = {image["id"]: image for image in document.get("images", [])}
    names = skeleton.by_index()

    boxes: list[LabelledBox] = []
    keypoints: list[LabelledKeypoint] = []
    sequences: set[str] = set()

    for annotation in document.get("annotations", []):
        if join_on == "file_name":
            file_name = annotation.get("file_name")
        else:
            image = by_image_id.get(annotation.get("image_id"))
            file_name = image.get("file_name") if image else None
        if not file_name:
            continue

        found = _FRAME.match(file_name)
        if not found:
            continue
        sequence = found.group("sequence")
        frame_index = int(found.group("frame")) + frame_offset
        if frame_index < 0:
            continue
        sequences.add(sequence)

        source_id = f"{dataset_name}/{sequence}"
        track_id = str(annotation.get("instance_id", ""))

        bbox = annotation.get("bbox")
        if bbox and len(bbox) == 4:
            x, y, width, height = (float(v) for v in bbox)
            boxes.append(
                LabelledBox(
                    frame_index=frame_index,
                    box=BoundingBox(x, y, x + width, y + height),
                    track_id=track_id,
                    animal_id=track_id,
                    source_id=source_id,
                )
            )

        raw = annotation.get("keypoints") or []
        if len(raw) != len(names) * 3:
            continue
        for definition, offset in zip(names, range(0, len(raw), 3), strict=True):
            x, y, flag = (float(v) for v in raw[offset : offset + 3])
            if flag == _NOT_LABELLED:
                # No annotation at all for this keypoint. Recording it as
                # "present but invisible" would invent a label.
                continue
            keypoints.append(
                LabelledKeypoint(
                    frame_index=frame_index,
                    name=definition.name,
                    x=x,
                    y=y,
                    # The release marks 2 visible and 1 occluded. Only the
                    # former is a position anything should be scored against.
                    visible=flag == _VISIBLE,
                    track_id=track_id,
                    source_id=source_id,
                    animal_id=track_id,
                )
            )

    return CocoKeypointLabels(
        boxes=tuple(boxes),
        keypoints=tuple(keypoints),
        split=path.stem,
        sequences=tuple(sorted(sequences)),
    )


class LabelledPoseBackend:
    """Serves annotated keypoints, matched to whichever instance a detection hit.

    A frame may hold several animals, so the detection has to be associated with
    one labelled instance before its keypoints mean anything. Association is by
    overlap, and a detection that matches nothing yields no keypoints rather
    than the nearest animal's.

    Its identity names these as labels, so no report can present them as a pose
    model's output.
    """

    def __init__(
        self,
        labels: CocoKeypointLabels,
        *,
        source: str = "dataset-keypoint-label",
        version: str = "1",
        min_iou: float = 0.30,
    ) -> None:
        self._source = source
        self._version = version
        self._min_iou = min_iou

        self._boxes: dict[tuple[str, int], list[tuple[str, BoundingBox]]] = {}
        for box in labels.boxes:
            self._boxes.setdefault((box.source_id, box.frame_index), []).append(
                (box.track_id, box.box)
            )
        self._keypoints: dict[tuple[str, int, str], list[LabelledKeypoint]] = {}
        for keypoint in labels.keypoints:
            self._keypoints.setdefault(
                (keypoint.source_id, keypoint.frame_index, keypoint.track_id), []
            ).append(keypoint)

    @property
    def model_identity(self) -> str:
        return f"{self._source}@{self._version}"

    @property
    def native_skeleton(self) -> str:
        return "profile"

    @property
    def keypoint_map(self) -> dict[str, str]:
        return {}

    def estimate(self, image, detection):
        from ..perception.pose import NativeKeypoint

        key = (detection.provenance.source_id, detection.provenance.frame_index)
        candidates = self._boxes.get(key, ())
        best_track, best_overlap = "", 0.0
        for track_id, box in candidates:
            overlap = detection.box.iou(box)
            if overlap > best_overlap:
                best_track, best_overlap = track_id, overlap
        if not best_track or best_overlap < self._min_iou:
            return []

        return [
            NativeKeypoint(
                name=keypoint.name,
                x=keypoint.x,
                y=keypoint.y,
                # Occluded keypoints are labelled but were not seen. Handing
                # them over at full confidence would present a position nobody
                # observed as an observation.
                confidence=1.0 if keypoint.visible else 0.0,
            )
            for keypoint in self._keypoints.get((*key, best_track), ())
        ]
