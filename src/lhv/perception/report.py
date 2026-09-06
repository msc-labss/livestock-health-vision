"""The perception run summary.

Low-confidence results stay in the stream. This is where they are counted, so
that "the run produced 4,000 detections" is never quietly a different statement
from "the run produced 4,000 detections it stood behind".

The same reasoning covers placeholder weights. Output from weights not trained
for this task measures the placeholder rather than an achievable result, and a
report that does not say so invites the number to be read as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PerceptionReport"]


@dataclass
class PerceptionReport:
    source_id: str = ""
    detector_identity: str = ""
    pose_identity: str = ""
    skeleton: str = ""

    frames_processed: int = 0
    frames_without_detection: int = 0

    detections_emitted: int = 0
    detections_low_confidence: int = 0

    tracklets_formed: int = 0
    terminations: dict[str, int] = field(default_factory=dict)

    poses_emitted: int = 0
    poses_low_confidence: int = 0
    keypoints_emitted: int = 0
    keypoints_not_visible: int = 0

    # role -> reason, for every weight the profile declares a placeholder.
    placeholder_weights: dict[str, str] = field(default_factory=dict)

    def describe(self) -> str:
        lines = [
            f"perception over {self.source_id}",
            f"  detector: {self.detector_identity}",
            f"  pose: {self.pose_identity} against skeleton {self.skeleton}",
            f"  frames: {self.frames_processed} "
            f"({self.frames_without_detection} with no detection)",
            f"  detections: {self.detections_emitted} "
            f"({self.detections_low_confidence} marked low confidence)",
            f"  tracklets: {self.tracklets_formed} "
            f"[{', '.join(f'{k}={v}' for k, v in sorted(self.terminations.items()))}]",
            f"  poses: {self.poses_emitted} ({self.poses_low_confidence} marked low confidence)",
            f"  keypoints: {self.keypoints_emitted} ({self.keypoints_not_visible} not visible)",
        ]
        for role, reason in sorted(self.placeholder_weights.items()):
            lines.append(
                f"  PLACEHOLDER: the {role} weights are declared a placeholder"
                + (f" ({reason})" if reason else "")
                + f". Any {role} metric here measures the placeholder, not an achievable result."
            )
        return "\n".join(lines)
