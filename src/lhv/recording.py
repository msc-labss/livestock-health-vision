"""Checking a candidate recording against the P1 recording specification.

The specification in docs/P1-RECORDING-SPECIFICATION.md states requirements that
decide whether a measurement is possible at all — a stride sampled below its own
band is not measured, an identity never recorded cannot be recovered. Each of
them was derived from something P0 measured on real public data, and each comes
with a test. This module is those tests, so a farm's pilot recording can be
judged before the study starts rather than after.

It reports what it measured alongside the verdict, and says "unknown" where it
cannot tell, because a check that quietly passes when it could not run is worse
than no check.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import ResolvedConfig
from .ingest.source import register_source
from .ingest.stream import Ingestor
from .perception.detect import Detector
from .perception.track import Tracker
from .profiles import SpeciesProfile

__all__ = ["Requirements", "Check", "RecordingReport", "check_recording"]

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"


@dataclass(frozen=True)
class Requirements:
    """The thresholds the specification states, in one place and declared.

    Every default is traceable to a P0 measurement recorded in the
    specification; none of them is a round number chosen for looking tidy.
    """

    # R2: samples per stride at the fast end of the profile's prior band.
    min_frame_rate: float = 15.0
    preferred_frame_rate: float = 25.0
    # R3: a lane that admits one animal at a time.
    min_single_animal_share: float = 0.95
    # R1: below this share of frames carrying a detection, the tool cannot tell a
    # sparsely occupied lane from a view the detector was never trained on, and
    # says so rather than guessing.
    min_detected_share: float = 0.10
    # R4: framing, as the animal's long side over the frame's long side.
    min_size_share: float = 0.25
    max_size_share: float = 0.50
    min_transit_seconds: float = 3.0
    # How much of the file to analyse for transit, which needs continuity.
    segment_seconds: float = 60.0


@dataclass(frozen=True)
class Check:
    requirement: str
    name: str
    verdict: str
    measured: str
    expected: str
    note: str = ""

    @property
    def symbol(self) -> str:
        return {PASS: "ok", FAIL: "FAIL", UNKNOWN: "?"}[self.verdict]


@dataclass
class RecordingReport:
    path: str
    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == FAIL]

    @property
    def unknown(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == UNKNOWN]

    @property
    def ok(self) -> bool:
        return not self.failed

    def describe(self) -> str:
        lines = [f"recording check: {self.path}", ""]
        width = max((len(c.name) for c in self.checks), default=0)
        for check in self.checks:
            lines.append(
                f"  [{check.symbol:>4}] {check.requirement} {check.name:<{width}}  "
                f"measured {check.measured}  (expected {check.expected})"
            )
            if check.note:
                lines.append(f"         {check.note}")
        lines.append("")
        if self.failed:
            lines.append(
                f"{len(self.failed)} requirement(s) not met. Each one makes some "
                f"measurement impossible rather than merely noisy."
            )
        else:
            lines.append("every requirement this tool can judge is met.")
        if self.unknown:
            lines.append(
                f"{len(self.unknown)} requirement(s) could not be judged here and need the "
                f"manual step the specification describes."
            )
        return "\n".join(lines)


def _container_timestamp(path: Path) -> tuple[datetime | None, str]:
    """The capture time a container records, if it records one."""
    if shutil.which("ffprobe") is None:
        return None, "ffprobe not available, so container metadata was not read"
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_entries",
                "format_tags=creation_time",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        tags = json.loads(result.stdout or "{}").get("format", {}).get("tags", {})
        raw = tags.get("creation_time")
        if not raw:
            return None, "the container records no creation_time tag"
        return datetime.fromisoformat(raw.replace("Z", "+00:00")), ""
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"container metadata could not be read: {exc}"


def check_recording(
    path: str | Path,
    profile: SpeciesProfile,
    config: ResolvedConfig,
    *,
    detector_backend=None,
    requirements: Requirements | None = None,
) -> RecordingReport:
    """Judge one pilot recording against the requirements it can be judged on."""
    import cv2

    path = Path(path)
    requirements = requirements or Requirements()
    report = RecordingReport(path=str(path))

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open {path}")
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    # -- R2 frame rate ------------------------------------------------------
    report.checks.append(
        Check(
            requirement="R2",
            name="frame rate",
            verdict=PASS if frame_rate >= requirements.min_frame_rate else FAIL,
            measured=f"{frame_rate:.3g} fps",
            expected=f">= {requirements.min_frame_rate:g} fps",
            note=(
                ""
                if frame_rate >= requirements.preferred_frame_rate
                else f"{requirements.preferred_frame_rate:g} fps is preferred; at "
                f"{frame_rate:.3g} fps a stride at the fast end of the profile's prior band "
                f"gets {frame_rate / 2.5:.1f} samples"
            ),
        )
    )

    # -- R6 capture time ----------------------------------------------------
    stamp, why = _container_timestamp(path)
    report.checks.append(
        Check(
            requirement="R6",
            name="capture time",
            verdict=PASS if stamp else FAIL,
            measured=stamp.isoformat() if stamp else "absent",
            expected="a creation time in the container",
            note=why,
        )
    )

    # -- detection-derived checks -------------------------------------------
    if detector_backend is None:
        for requirement, name in (
            ("R1", "animal visibility"),
            ("R3", "one animal at a time"),
            ("R4", "framing"),
            ("R4", "transit duration"),
        ):
            report.checks.append(
                Check(
                    requirement=requirement,
                    name=name,
                    verdict=UNKNOWN,
                    measured="not attempted",
                    expected="see the specification",
                    note="no detector was supplied, so nothing in the frames was measured",
                )
            )
        return report

    source = register_source(
        source_id=f"pilot/{path.name}",
        camera_id="pilot-camera",
        site_key="pilot-site",
        animal_set_key="pilot-herd",
        media_path=str(path),
        day_key="pilot",
        kind="video",
    )
    detector = Detector(detector_backend, config)
    limit = int(requirements.segment_seconds * frame_rate) if frame_rate > 0 else frames
    records = []
    for frame in Ingestor(source, config).iter_frames():
        records.append(detector.detect_frame(frame))
        frame.image = None
        if len(records) >= limit:
            break

    populated = [r for r in records if not r.is_empty]
    detected_share = len(populated) / len(records) if records else 0.0
    if not populated:
        visibility, note = (
            FAIL,
            "a pretrained detector finding nothing at all is the signature of a view outside "
            "its training distribution, which is exactly what a straight-down mount produced "
            "in P0",
        )
    elif detected_share < requirements.min_detected_share:
        visibility, note = (
            UNKNOWN,
            "too few frames carry a detection to tell a sparsely occupied lane from a mount "
            "the detector cannot read. The specification's labelling test settles it: label "
            "200 frames and require paws visible in 60% of instances",
        )
    else:
        visibility, note = PASS, ""
    report.checks.append(
        Check(
            requirement="R1",
            name="animal visibility",
            verdict=visibility,
            measured=f"{len(populated)}/{len(records)} frames ({detected_share:.1%})",
            expected=f">= {requirements.min_detected_share:.0%} of frames, or a labelling check",
            note=note,
        )
    )

    if populated:
        singles = sum(1 for r in populated if len(r) == 1)
        share_single = singles / len(populated)
        report.checks.append(
            Check(
                requirement="R3",
                name="one animal at a time",
                verdict=PASS if share_single >= requirements.min_single_animal_share else FAIL,
                measured=f"{share_single:.1%} of populated frames hold exactly one",
                expected=f">= {requirements.min_single_animal_share:.0%}",
                note=(
                    ""
                    if share_single >= requirements.min_single_animal_share
                    else "an external identifier cannot say which animal is which when several "
                    "share a window; P0 left 507 of 761 tracklets unresolved for this reason"
                ),
            )
        )

        long_side = max(width, height) or 1
        sizes = [
            max(d.box.width, d.box.height) / long_side for r in populated for d in r.detections
        ]
        sizes.sort()
        median_size = sizes[len(sizes) // 2] if sizes else 0.0
        within = requirements.min_size_share <= median_size <= requirements.max_size_share
        report.checks.append(
            Check(
                requirement="R4",
                name="framing",
                verdict=PASS if within else FAIL,
                measured=f"median animal spans {median_size:.1%} of the frame's long side",
                expected=f"{requirements.min_size_share:.0%}-{requirements.max_size_share:.0%}",
            )
        )

        tracker = Tracker(config, source_id=source.source_id)
        tracklets = tracker.track(records)
        if tracklets and frame_rate > 0:
            transits = sorted(
                (t.last_frame_index - t.first_frame_index + 1) / frame_rate for t in tracklets
            )
            median_transit = transits[len(transits) // 2]
            report.checks.append(
                Check(
                    requirement="R4",
                    name="transit duration",
                    verdict=(PASS if median_transit >= requirements.min_transit_seconds else FAIL),
                    measured=f"median {median_transit:.1f} s over {len(tracklets)} tracklet(s)",
                    expected=f">= {requirements.min_transit_seconds:g} s",
                    note=(
                        f"at {frame_rate:.3g} fps that is "
                        f"{median_transit * frame_rate:.0f} frames per pass"
                    ),
                )
            )
        else:
            report.checks.append(
                Check(
                    requirement="R4",
                    name="transit duration",
                    verdict=UNKNOWN,
                    measured="no tracklet formed",
                    expected=f">= {requirements.min_transit_seconds:g} s",
                )
            )
    else:
        for requirement, name in (("R3", "one animal at a time"), ("R4", "framing")):
            report.checks.append(
                Check(
                    requirement=requirement,
                    name=name,
                    verdict=UNKNOWN,
                    measured="nothing detected",
                    expected="see the specification",
                    note="cannot be judged until R1 passes",
                )
            )

    report.checks.append(
        Check(
            requirement="R5",
            name="identity feed",
            verdict=UNKNOWN,
            measured="not attempted",
            expected="one identifier per pass, persistent across days",
            note="join a day of the identifier feed against the footage, as the specification says",
        )
    )
    return report
