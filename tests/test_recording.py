"""Checking a pilot recording against the P1 recording specification."""

from __future__ import annotations

import pytest

from lhv.perception import BoundingBox
from lhv.perception.detect import _RawDetection
from lhv.profiles import load_profile
from lhv.recording import FAIL, PASS, UNKNOWN, Check, Requirements, check_recording

from .conftest import write_video


@pytest.fixture
def profile():
    return load_profile("cattle")


class _StubDetector:
    """A detector with a scripted answer, so the checks can be exercised."""

    def __init__(self, per_frame) -> None:
        self.per_frame = per_frame

    @property
    def model_identity(self) -> str:
        return "stub-detector@1"

    def detect(self, image, provenance=None):
        count, size = self.per_frame(provenance.frame_index if provenance else 0)
        width, height = image.shape[1], image.shape[0]
        side = size * max(width, height)
        return [
            _RawDetection(
                box=BoundingBox(20.0 + i * 5, 20.0, 20.0 + i * 5 + side, 20.0 + side * 0.4),
                label="animal",
                confidence=0.9,
            )
            for i in range(count)
        ]


def _check(video, profile, config, detector=None, **kwargs):
    return check_recording(
        video, profile, config, detector_backend=detector, requirements=Requirements(**kwargs)
    )


# -- R2 frame rate -----------------------------------------------------------


def test_a_slow_recording_fails_the_frame_rate_requirement(tmp_path, profile, config) -> None:
    """A stride sampled below its own band is not measured badly, but not at all."""
    video = write_video(tmp_path / "slow.avi", frames=40, fps=8.0)
    report = _check(video, profile, config)
    rate = next(c for c in report.checks if c.requirement == "R2")
    assert rate.verdict == FAIL
    assert "8 fps" in rate.measured
    assert "3.2 samples" in rate.note


def test_an_adequate_frame_rate_passes(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "fast.avi", frames=60, fps=25.0)
    rate = next(c for c in _check(video, profile, config).checks if c.requirement == "R2")
    assert rate.verdict == PASS
    assert rate.note == ""


# -- R6 capture time ---------------------------------------------------------


def test_a_recording_without_a_capture_time_fails(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "undated.avi", frames=20, fps=25.0)
    stamp = next(c for c in _check(video, profile, config).checks if c.requirement == "R6")
    assert stamp.verdict == FAIL
    assert stamp.measured == "absent"


# -- R1 animal visibility ----------------------------------------------------


def test_finding_nothing_at_all_fails(tmp_path, profile, config) -> None:
    """What a straight-down mount produced in P0."""
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (0, 0.0)))
    visibility = next(c for c in report.checks if c.requirement == "R1")
    assert visibility.verdict == FAIL
    assert "outside" in visibility.note


def test_finding_almost_nothing_is_reported_as_undecidable(tmp_path, profile, config) -> None:
    """A sparsely occupied lane and an unreadable mount look the same from here."""
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(
        video, profile, config, detector=_StubDetector(lambda i: (1, 0.35) if i < 2 else (0, 0.0))
    )
    visibility = next(c for c in report.checks if c.requirement == "R1")
    assert visibility.verdict == UNKNOWN
    assert "labelling test settles it" in visibility.note


def test_a_readable_mount_passes(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (1, 0.35)))
    assert next(c for c in report.checks if c.requirement == "R1").verdict == PASS


# -- R3 one animal at a time -------------------------------------------------


def test_a_lane_carrying_several_animals_fails(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (3, 0.30)))
    crowd = next(c for c in report.checks if c.requirement == "R3")
    assert crowd.verdict == FAIL
    assert "507 of 761" in crowd.note


def test_a_single_file_lane_passes(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (1, 0.35)))
    assert next(c for c in report.checks if c.requirement == "R3").verdict == PASS


# -- R4 framing and transit --------------------------------------------------


def test_an_animal_too_small_in_frame_fails(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (1, 0.05)))
    framing = next(c for c in report.checks if c.name == "framing")
    assert framing.verdict == FAIL


def test_correct_framing_passes(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (1, 0.35)))
    assert next(c for c in report.checks if c.name == "framing").verdict == PASS


def test_a_transit_too_brief_to_measure_fails(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=50, fps=25.0)
    # Present for four frames, then gone: half a second of animal.
    report = _check(
        video, profile, config, detector=_StubDetector(lambda i: (1, 0.35) if i < 4 else (0, 0.0))
    )
    transit = next((c for c in report.checks if c.name == "transit duration"), None)
    assert transit is not None
    assert transit.verdict in {FAIL, UNKNOWN}


# -- the report --------------------------------------------------------------


def test_requirements_needing_a_human_are_reported_as_such(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=20, fps=25.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (1, 0.35)))
    identity = next(c for c in report.checks if c.requirement == "R5")
    assert identity.verdict == UNKNOWN
    assert "identifier feed" in identity.note
    assert report.unknown


def test_without_a_detector_nothing_in_the_frames_is_claimed(tmp_path, profile, config) -> None:
    """A check that quietly passes when it could not run is worse than none."""
    video = write_video(tmp_path / "v.avi", frames=20, fps=25.0)
    report = _check(video, profile, config, detector=None)
    for requirement in ("R1", "R3", "R4"):
        checks = [c for c in report.checks if c.requirement == requirement]
        assert checks
        assert all(c.verdict == UNKNOWN for c in checks)


def test_the_report_says_which_requirements_were_not_met(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=40, fps=8.0)
    report = _check(video, profile, config, detector=_StubDetector(lambda i: (3, 0.30)))
    text = report.describe()
    assert not report.ok
    assert "FAIL" in text
    assert "impossible rather than merely noisy" in text
    assert all(c.requirement in text for c in report.checks)


def test_every_threshold_is_declared_rather_than_buried() -> None:
    """The specification's numbers live in one object, not scattered in the checks."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Requirements)}
    assert {
        "min_frame_rate",
        "preferred_frame_rate",
        "min_single_animal_share",
        "min_size_share",
        "max_size_share",
        "min_transit_seconds",
        "min_detected_share",
    } <= fields


def test_a_check_reports_what_it_measured_alongside_its_verdict() -> None:
    check = Check(
        requirement="R2", name="frame rate", verdict=FAIL, measured="8 fps", expected=">= 15 fps"
    )
    assert check.symbol == "FAIL"
    assert check.measured and check.expected


def test_the_specification_exists_and_states_every_requirement() -> None:
    from pathlib import Path

    spec = Path(__file__).resolve().parent.parent / "docs" / "P1-RECORDING-SPECIFICATION.md"
    assert spec.exists()
    text = spec.read_text(encoding="utf-8")
    for requirement in ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"):
        assert f"## {requirement} " in text, f"{requirement} is not stated in the specification"
    # Each requirement carries the evidence and the test.
    assert text.count("**Why.**") == 9
    assert text.count("**Test.**") == 9
