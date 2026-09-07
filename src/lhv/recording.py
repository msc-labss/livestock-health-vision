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
import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .config import ResolvedConfig
from .ingest.source import register_source
from .ingest.stream import Ingestor
from .perception.detect import Detector
from .perception.track import Tracker
from .profiles import SpeciesProfile

__all__ = [
    "Requirements",
    "Threshold",
    "Check",
    "Part",
    "RecordingReport",
    "check_recording",
]

PASS, FAIL, UNKNOWN = "pass", "fail", "unknown"

SPEC = "docs/P1-RECORDING-SPECIFICATION.md"


@dataclass(frozen=True)
class Threshold:
    """One bound, and where it came from.

    A number with no stated origin cannot be revised with confidence, because
    nothing says what would justify changing it. Two of the bounds below are
    disputed in the literature and held pending a paper this project has not
    been able to read; naming their source is what makes moving them an edit
    rather than an excavation.
    """

    value: float
    source: str

    def __str__(self) -> str:
        return f"{self.value:g} [{self.source}]"


@dataclass(frozen=True)
class Requirements:
    """The thresholds the specification states, in one place and attributed."""

    # R2. Held at the specification's figure: docs/P1-LITERATURE-FINDINGS.md
    # argues for 25 on foot-strike timing grounds, but that rests on a capture
    # rate in Kang et al. 2020 which has not been read. Moving it is one edit.
    min_frame_rate: Threshold = Threshold(15.0, f"{SPEC} R2")
    preferred_frame_rate: Threshold = Threshold(25.0, f"{SPEC} R2")
    # R3: a lane that admits one animal at a time.
    min_single_animal_share: Threshold = Threshold(0.95, f"{SPEC} R3")
    # R1: below this share of frames carrying a detection, the tool cannot tell a
    # sparsely occupied lane from a view the detector was never trained on, and
    # says so rather than guessing.
    min_detected_share: Threshold = Threshold(0.10, f"{SPEC} R1")
    # R4: framing, as the animal's long side over the frame's long side.
    min_size_share: Threshold = Threshold(0.25, f"{SPEC} R4")
    max_size_share: Threshold = Threshold(0.50, f"{SPEC} R4")
    min_transit_seconds: Threshold = Threshold(3.0, f"{SPEC} R4")
    # A transit must also cross the measured section, not merely last. Expressed
    # as a share of the frame's long side, so it needs no calibration.
    min_transit_displacement_share: Threshold = Threshold(0.4, f"{SPEC} R4")

    # Coverage of the recording. Contiguous windows because tracklet formation
    # needs consecutive frames; several of them because lane occupancy is not
    # uniform and an empty opening minute is a fact about when recording
    # started rather than about the recording.
    window_seconds: float = 30.0
    window_count: int = 4

    def frame_rate_floor(self, profile: SpeciesProfile) -> Threshold:
        """The stricter of the specification's floor and the profile's needs.

        Only features the profile declares *available* count. An unavailable
        feature's sampling requirement describes a measurement this
        configuration cannot make, and holding a recording to it would reject
        footage for a property of the backend.
        """
        floor = self.min_frame_rate
        for feature in profile.feature_set.available:
            required = getattr(feature, "requires_sampling_hz", 0.0) or 0.0
            if required > floor.value:
                floor = Threshold(
                    required,
                    f"profile feature {feature.name!r} requires_sampling_hz",
                )
        return floor


@dataclass(frozen=True)
class Part:
    """One part of a requirement's test, and whether it could be attempted."""

    name: str
    attempted: bool
    passed: bool = False
    detail: str = ""


@dataclass(frozen=True)
class Check:
    requirement: str
    name: str
    verdict: str
    measured: str
    expected: str
    note: str = ""
    # The parts of this requirement's test. A requirement whose test has several
    # parts reports which of them ran, because a verdict that averages an
    # attempted part with an unattempted one is not a verdict.
    parts: tuple[Part, ...] = ()

    @property
    def symbol(self) -> str:
        return {PASS: "ok", FAIL: "FAIL", UNKNOWN: "?"}[self.verdict]

    @property
    def unattempted(self) -> tuple[Part, ...]:
        return tuple(p for p in self.parts if not p.attempted)

    @classmethod
    def from_parts(
        cls,
        requirement: str,
        name: str,
        parts: Sequence[Part],
        *,
        measured: str,
        expected: str,
        note: str = "",
    ) -> Check:
        """Combine part results into one verdict.

        A failed part fails the check. Otherwise any unattempted part makes it
        unknown, however well the attempted parts went: a check that quietly
        passes when it could not run is worse than no check.
        """
        parts = tuple(parts)
        if any(p.attempted and not p.passed for p in parts):
            verdict = FAIL
        elif any(not p.attempted for p in parts):
            verdict = UNKNOWN
        else:
            verdict = PASS
        return cls(
            requirement=requirement,
            name=name,
            verdict=verdict,
            measured=measured,
            expected=expected,
            note=note,
            parts=parts,
        )


@dataclass
class RecordingReport:
    path: str
    checks: list[Check] = field(default_factory=list)
    # What this verdict was reached under. The same recording is acceptable
    # against one geometry and not another, so a verdict without its
    # configuration cannot be interpreted later.
    profile_id: str = ""
    feature_set_version: str = ""
    view: str = ""
    # How much of the recording the frame-derived checks saw.
    analysed_seconds: float = 0.0
    duration_seconds: float = 0.0

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == FAIL]

    @property
    def unknown(self) -> list[Check]:
        return [c for c in self.checks if c.verdict == UNKNOWN]

    @property
    def ok(self) -> bool:
        """True when nothing failed. Unjudged requirements do not fail a run.

        They are reported separately instead: an unattempted check is not
        evidence against the recording, it is a gap in what was checked.
        """
        return not self.failed

    def describe(self) -> str:
        lines = [f"recording check: {self.path}"]
        lines.append(
            f"  judged against profile {self.profile_id or 'unknown'}, "
            f"feature set v{self.feature_set_version or '?'}, "
            f"{self.view or 'unspecified'} view"
        )
        if self.duration_seconds > 0:
            share = self.analysed_seconds / self.duration_seconds
            lines.append(
                f"  analysed {self.analysed_seconds:.0f} s of "
                f"{self.duration_seconds:.0f} s ({share:.0%})"
            )
        lines.append("")

        width = max((len(c.name) for c in self.checks), default=0)
        for check in self.checks:
            lines.append(
                f"  [{check.symbol:>4}] {check.requirement} {check.name:<{width}}  "
                f"measured {check.measured}  (expected {check.expected})"
            )
            for part in check.parts:
                mark = "ok" if part.passed else "FAIL"
                mark = mark if part.attempted else "not attempted"
                lines.append(
                    f"         - {part.name}: {mark}" + (f" — {part.detail}" if part.detail else "")
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
            lines.append("no requirement this tool judged was unmet.")
        if self.unknown:
            lines.append(
                f"{len(self.unknown)} requirement(s) could not be judged here. That is not "
                f"a pass: each needs the manual step the specification describes."
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


_FILENAME_TIME_PATTERNS = (
    # 2024-03-05T13-04-28, 2024-03-05T13:04:28, 2024-03-05_130428
    r"(?P<Y>\d{4})[-_]?(?P<m>\d{2})[-_]?(?P<d>\d{2})[T_ -]"
    r"(?P<H>\d{2})[-:_]?(?P<M>\d{2})[-:_]?(?P<S>\d{2})",
)


def _filename_timestamp(path: Path) -> tuple[datetime | None, str]:
    """A capture time carried in the filename, which R6 permits as a fallback."""
    for pattern in _FILENAME_TIME_PATTERNS:
        match = re.search(pattern, path.name)
        if not match:
            continue
        g = match.groupdict()
        try:
            return (
                datetime(
                    int(g["Y"]),
                    int(g["m"]),
                    int(g["d"]),
                    int(g["H"]),
                    int(g["M"]),
                    int(g["S"]),
                ),
                "",
            )
        except ValueError as exc:
            return None, f"the filename looks like a timestamp but is not one: {exc}"
    return None, "the filename carries no recognisable timestamp"


def _frame_rate_is_constant(path: Path) -> tuple[bool | None, str]:
    """Whether the container reports a constant frame rate.

    Compares the stream's average and nominal rates: a container that declares
    both and disagrees with itself is variable-rate. This is weaker than reading
    packet timestamps and it is what is available without decoding the file, so
    the limit is reported rather than glossed.
    """
    if shutil.which("ffprobe") is None:
        return None, "ffprobe not available, so frame-rate constancy was not read"
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        streams = json.loads(result.stdout or "{}").get("streams", [])
        if not streams:
            return None, "the container declares no video stream rates"
        stream = streams[0]
        avg, nominal = stream.get("avg_frame_rate"), stream.get("r_frame_rate")
        if not avg or not nominal or "0/0" in (avg, nominal):
            return None, "the container declares only one of the two frame rates"

        def ratio(text: str) -> float | None:
            num, _, den = text.partition("/")
            try:
                d = float(den or 1)
                return float(num) / d if d else None
            except ValueError:
                return None

        a, n = ratio(avg), ratio(nominal)
        if a is None or n is None:
            return None, "the container's frame rates could not be parsed"
        if n == 0:
            return None, "the container declares a zero nominal frame rate"
        constant = abs(a - n) / n <= 0.01
        detail = f"average {a:.4g} fps against nominal {n:.4g} fps"
        return constant, detail
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return None, f"frame-rate constancy could not be read: {exc}"


def _window_bounds(
    frames: int, frame_rate: float, requirements: Requirements
) -> list[tuple[int, int]]:
    """Contiguous frame windows spread evenly across the recording.

    Contiguous so tracklets still form; spread so an empty opening minute is not
    mistaken for an empty recording. A file shorter than one window is analysed
    whole rather than sampled, because there is nothing to spread.
    """
    if frames <= 0 or frame_rate <= 0:
        return [(0, max(frames, 0))]
    window = max(int(requirements.window_seconds * frame_rate), 1)
    count = max(requirements.window_count, 1)
    if frames <= window * count:
        return [(0, frames)]
    stride = frames // count
    bounds = []
    for i in range(count):
        start = i * stride
        bounds.append((start, min(start + window, frames)))
    return bounds


def _transit_check(
    records,
    source_id: str,
    config: ResolvedConfig,
    frame_rate: float,
    long_side: int,
    requirements: Requirements,
) -> Check:
    """R4's transit, measured as a traverse rather than as time in frame.

    An animal that halts in the lane persists indefinitely and produces no
    strides; a tracklet that fragments gives a short persistence from a complete
    traverse. Neither is what the requirement is about, so displacement decides
    what counts as a transit and duration is reported for the ones that do.
    """
    import numpy as np

    minimum = requirements.min_transit_seconds
    span = requirements.min_transit_displacement_share

    if frame_rate <= 0:
        return Check(
            requirement="R4",
            name="transit",
            verdict=UNKNOWN,
            measured="no frame rate",
            expected=f">= {minimum.value:g} s [{minimum.source}]",
            note="duration cannot be derived without a frame rate",
        )

    tracklets = Tracker(config, source_id=source_id).track(records)
    if not tracklets:
        return Check(
            requirement="R4",
            name="transit",
            verdict=UNKNOWN,
            measured="no tracklet formed",
            expected=f">= {minimum.value:g} s [{minimum.source}]",
        )

    expected = f">= {minimum.value:g} s over >= {span.value:.0%} of frame [{span.source}]"

    transits, stalled = [], 0
    for tracklet in tracklets:
        centres = np.array([d.box.centre for d in tracklet.detections], dtype=float)
        if len(centres) < 2:
            stalled += 1
            continue
        displacement = float(np.linalg.norm(centres[-1] - centres[0])) / long_side
        seconds = (tracklet.last_frame_index - tracklet.first_frame_index + 1) / frame_rate
        if displacement >= span.value:
            transits.append((seconds, displacement))
        else:
            stalled += 1

    if not transits:
        return Check(
            requirement="R4",
            name="transit",
            verdict=FAIL,
            measured=f"0 traverses; {stalled} tracklet(s) persisted without crossing",
            expected=expected,
            note=(
                "persistence is not traversal: an animal that halts in the lane stays in "
                "frame and produces no strides, which is a finding about the lane rather "
                "than a measurement to discard"
            ),
        )

    transits.sort()
    median_seconds, median_span = transits[len(transits) // 2]
    note = f"at {frame_rate:.4g} fps that is {median_seconds * frame_rate:.0f} frames per pass"
    if stalled:
        note += f"; {stalled} tracklet(s) persisted without crossing the section"
    return Check(
        requirement="R4",
        name="transit",
        verdict=PASS if median_seconds >= minimum.value else FAIL,
        measured=(
            f"median {median_seconds:.1f} s over {median_span:.0%} of the frame, "
            f"{len(transits)} traverse(s)"
        ),
        expected=expected,
        note=note,
    )


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
    report = RecordingReport(
        path=str(path),
        profile_id=f"{profile.skeleton.identifier}@{profile.skeleton.version}",
        feature_set_version=profile.feature_set.version,
        view=profile.skeleton.view,
    )

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"could not open {path}")
    frame_rate = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()

    duration = frames / frame_rate if frame_rate > 0 else 0.0
    report.duration_seconds = duration

    # -- R2 frame rate: sufficient, and constant ----------------------------
    floor = requirements.frame_rate_floor(profile)
    preferred = requirements.preferred_frame_rate
    constant, constancy_detail = _frame_rate_is_constant(path)
    rate_note = ""
    if 0 < frame_rate < preferred.value:
        rate_note = f"{preferred} fps is preferred"
    report.checks.append(
        Check.from_parts(
            "R2",
            "frame rate",
            [
                Part(
                    "rate",
                    attempted=frame_rate > 0,
                    passed=frame_rate >= floor.value,
                    detail=f"{frame_rate:.4g} fps against a floor of {floor}",
                ),
                Part(
                    "constant",
                    attempted=constant is not None,
                    passed=bool(constant),
                    detail=constancy_detail,
                ),
            ],
            measured=f"{frame_rate:.4g} fps",
            expected=f">= {floor.value:g} fps, constant [{floor.source}]",
            note=rate_note,
        )
    )

    # -- R6 capture time ----------------------------------------------------
    stamp, why = _container_timestamp(path)
    time_source = "container"
    if stamp is None:
        stamp, filename_why = _filename_timestamp(path)
        if stamp is not None:
            time_source = "filename"
        else:
            why = f"{why}; {filename_why}"
    report.checks.append(
        Check.from_parts(
            "R6",
            "capture time",
            [
                Part(
                    "a readable capture time",
                    attempted=True,
                    passed=stamp is not None,
                    detail=(f"read from the {time_source}" if stamp else why),
                ),
                # The other half of R6's test compares the video clock against
                # the identity feed's. No feed is supplied here, so it is not
                # attempted — and the requirement therefore cannot report a pass.
                Part(
                    "offset to the identity feed under 1 s",
                    attempted=False,
                    detail="no identity feed was supplied; join one day of it as R6 describes",
                ),
            ],
            measured=stamp.isoformat() if stamp else "absent",
            expected="a capture time, and a clock within 1 s of the identity feed",
        )
    )

    # -- detection-derived checks -------------------------------------------
    if detector_backend is None:
        for requirement, name in (
            ("R1", "animal visibility"),
            ("R3", "one animal at a time"),
            ("R4", "framing"),
            ("R4", "transit"),
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
        view=profile.skeleton.view,
    )
    detector = Detector(detector_backend, config)

    # Contiguous windows spread across the recording. Contiguous because
    # tracklet formation needs consecutive frames; spread because lane occupancy
    # is not uniform and an empty opening minute says nothing about the rest.
    windows = _window_bounds(frames, frame_rate, requirements)
    wanted = {index for start, stop in windows for index in range(start, stop)}
    report.analysed_seconds = (len(wanted) / frame_rate) if frame_rate > 0 else 0.0

    records = []
    for index, frame in enumerate(Ingestor(source, config).iter_frames()):
        if index in wanted:
            records.append(detector.detect_frame(frame))
        frame.image = None
        if index >= max(wanted, default=-1):
            break

    populated = [r for r in records if not r.is_empty]
    detected_share = len(populated) / len(records) if records else 0.0
    minimum = requirements.min_detected_share
    if not populated:
        visibility, note = (
            FAIL,
            "a pretrained detector finding nothing at all is the signature of a view outside "
            "its training distribution, which is exactly what a straight-down mount produced "
            "in P0",
        )
    elif detected_share < minimum.value:
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
            expected=f">= {minimum.value:.0%} of frames [{minimum.source}], or a labelling check",
            note=note,
        )
    )

    if not populated:
        for requirement, name in (
            ("R3", "one animal at a time"),
            ("R4", "framing"),
            ("R4", "transit"),
        ):
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
        return report

    singles = sum(1 for r in populated if len(r) == 1)
    share_single = singles / len(populated)
    single_min = requirements.min_single_animal_share
    report.checks.append(
        Check(
            requirement="R3",
            name="one animal at a time",
            verdict=PASS if share_single >= single_min.value else FAIL,
            measured=f"{share_single:.1%} of populated frames hold exactly one",
            expected=f">= {single_min.value:.0%} [{single_min.source}]",
            note=(
                ""
                if share_single >= single_min.value
                else "an external identifier cannot say which animal is which when several "
                "share a window; P0 left 507 of 761 tracklets unresolved for this reason"
            ),
        )
    )

    long_side = max(width, height) or 1
    sizes = sorted(
        max(d.box.width, d.box.height) / long_side for r in populated for d in r.detections
    )
    median_size = sizes[len(sizes) // 2] if sizes else 0.0
    lower, upper = requirements.min_size_share, requirements.max_size_share
    within = lower.value <= median_size <= upper.value
    report.checks.append(
        Check(
            requirement="R4",
            name="framing",
            verdict=PASS if within else FAIL,
            measured=f"median animal spans {median_size:.1%} of the frame's long side",
            expected=f"{lower.value:.0%}-{upper.value:.0%} [{lower.source}]",
        )
    )

    # -- R4 transit: displacement, not persistence --------------------------
    report.checks.append(
        _transit_check(records, source.source_id, config, frame_rate, long_side, requirements)
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
