"""Checking a pilot recording against the P1 recording specification."""

from __future__ import annotations

import pytest

from lhv.perception import BoundingBox
from lhv.perception.detect import _RawDetection
from lhv.profiles import load_profile
from lhv.recording import (
    FAIL,
    PASS,
    UNKNOWN,
    Check,
    Part,
    Requirements,
    Threshold,
    check_recording,
)

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
    # The rate part fails by name, and the floor it failed against is attributed.
    failing = [p for p in rate.parts if p.attempted and not p.passed]
    assert [p.name for p in failing] == ["rate"]
    assert "floor of 25" in failing[0].detail
    assert "P1-RECORDING-SPECIFICATION" in rate.expected


def test_an_adequate_frame_rate_passes_the_rate_part(tmp_path, profile, config) -> None:
    """Asserted on the rate part, not the whole check, and deliberately so.

    R2 has two parts and the second needs ffprobe to read the container's two
    declared rates. On a machine without it that part is unattempted and the
    whole check is unknown, which is the correct answer and not a fact about the
    frame rate. Asserting the verdict here would pass or fail on whether ffprobe
    happens to be installed — which is how this test failed in CI and not
    locally.
    """
    # At the floor exactly: the rate part passes and the note says the preferred
    # rate is higher, which is advice rather than a failure.
    at_floor = write_video(tmp_path / "floor.avi", frames=60, fps=25.0)
    rate = next(c for c in _check(at_floor, profile, config).checks if c.requirement == "R2")
    part = next(p for p in rate.parts if p.name == "rate")
    assert part.attempted and part.passed
    assert rate.verdict in {PASS, UNKNOWN}
    assert "preferred" in rate.note

    # At the preferred rate there is nothing left to advise.
    preferred = write_video(tmp_path / "preferred.avi", frames=60, fps=50.0)
    rate = next(c for c in _check(preferred, profile, config).checks if c.requirement == "R2")
    assert next(p for p in rate.parts if p.name == "rate").passed
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
    transit = next((c for c in report.checks if c.name == "transit"), None)
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
    requirements = ("R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10")
    for requirement in requirements:
        assert f"## {requirement} " in text, f"{requirement} is not stated in the specification"
    # Each requirement carries the evidence and the test. Counted rather than
    # hard-coded, so adding a requirement without either one fails here.
    assert text.count("**Why.**") == len(requirements)
    assert text.count("**Test.**") == len(requirements)


# -- 1. thresholds carry their provenance ------------------------------------


def test_the_specification_floor_applies_when_the_profile_declares_nothing(profile) -> None:
    """Feature-set version 1 declares no sampling requirement at all."""
    assert all(f.requires_sampling_hz == 0.0 for f in profile.feature_set.available)
    floor = Requirements().frame_rate_floor(profile)
    assert floor.value == 25.0
    assert "P1-RECORDING-SPECIFICATION" in floor.source


def test_a_stricter_profile_requirement_wins_and_is_named(profile) -> None:
    import dataclasses

    features = tuple(
        dataclasses.replace(f, requires_sampling_hz=40.0) if f.name == "speed" else f
        for f in profile.feature_set.features
    )
    demanding = dataclasses.replace(
        profile, feature_set=dataclasses.replace(profile.feature_set, features=features)
    )
    floor = Requirements().frame_rate_floor(demanding)
    assert floor.value == 40.0
    assert "speed" in floor.source


def test_an_unavailable_features_requirement_is_ignored(profile) -> None:
    """It describes a measurement this configuration cannot make."""
    import dataclasses

    unavailable = profile.feature_set.unavailable[0].name
    features = tuple(
        dataclasses.replace(f, requires_sampling_hz=99.0) if f.name == unavailable else f
        for f in profile.feature_set.features
    )
    loaded = dataclasses.replace(
        profile, feature_set=dataclasses.replace(profile.feature_set, features=features)
    )
    assert Requirements().frame_rate_floor(loaded).value == 25.0


def test_every_threshold_names_where_it_came_from() -> None:
    requirements = Requirements()
    thresholds = [v for v in vars(requirements).values() if isinstance(v, Threshold)]
    assert thresholds
    for threshold in thresholds:
        assert threshold.source.strip(), f"{threshold} states no source"


# -- 2. incompleteness is visible --------------------------------------------


def test_a_partially_attempted_check_never_passes() -> None:
    check = Check.from_parts(
        "R6",
        "capture time",
        [Part("read a time", True, True), Part("clock offset", False, detail="no feed")],
        measured="m",
        expected="e",
    )
    assert check.verdict == UNKNOWN
    assert [p.name for p in check.unattempted] == ["clock offset"]


def test_a_failing_part_fails_the_check_even_beside_an_unattempted_one() -> None:
    check = Check.from_parts(
        "R2",
        "frame rate",
        [Part("rate", True, False), Part("constant", False)],
        measured="m",
        expected="e",
    )
    assert check.verdict == FAIL


def test_capture_time_alone_no_longer_reports_a_pass(tmp_path, profile, config) -> None:
    """R6's other half compares clocks against a feed this tool is never given."""
    video = write_video(tmp_path / "2024-03-05T13-04-28.avi", frames=30, fps=25.0)
    r6 = next(c for c in _check(video, profile, config).checks if c.requirement == "R6")
    assert r6.verdict == UNKNOWN
    assert any("identity feed" in p.name for p in r6.unattempted)


def test_unjudged_requirements_do_not_fail_the_run(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "2024-03-05T13-04-28.avi", frames=30, fps=25.0)
    report = _check(video, profile, config)
    assert report.unknown, "this recording has requirements no tool can judge"
    described = report.describe()
    assert "could not be judged" in described and "That is not a pass" in described


# -- 3. coverage of the recording --------------------------------------------


def test_a_recording_whose_opening_is_empty_is_still_judged(tmp_path, profile, config) -> None:
    """A lane is often empty at the top of the hour; that is not a finding."""
    video = write_video(tmp_path / "v.avi", frames=200, fps=25.0)
    report = _check(
        video,
        profile,
        config,
        detector=_StubDetector(lambda i: (1, 0.35) if i > 120 else (0, 0.0)),
        window_seconds=1.0,
        window_count=4,
    )
    visibility = next(c for c in report.checks if c.requirement == "R1")
    assert visibility.verdict is not FAIL, "the later material was never reached"


def test_the_report_states_how_much_it_analysed(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=200, fps=25.0)
    report = _check(
        video,
        profile,
        config,
        detector=_StubDetector(lambda i: (1, 0.35)),
        window_seconds=1.0,
        window_count=4,
    )
    assert 0 < report.analysed_seconds <= report.duration_seconds
    assert "analysed" in report.describe()


# -- 4. R2 judges constancy --------------------------------------------------


def test_constancy_is_a_separate_part_of_the_rate_check(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=60, fps=25.0)
    r2 = next(c for c in _check(video, profile, config).checks if c.requirement == "R2")
    assert [p.name for p in r2.parts] == ["rate", "constant"]
    assert "constant" in r2.expected


def test_constancy_that_cannot_be_established_is_not_assumed() -> None:
    check = Check.from_parts(
        "R2",
        "frame rate",
        [Part("rate", True, True), Part("constant", False, detail="ffprobe not available")],
        measured="25 fps",
        expected="e",
    )
    assert check.verdict == UNKNOWN, "an unreadable constancy must not read as constant"


# -- 5. R4 measures traversal ------------------------------------------------


def test_a_stationary_animal_does_not_satisfy_the_transit_requirement(
    tmp_path, profile, config
) -> None:
    """It stays in frame indefinitely and produces no strides."""
    video = write_video(tmp_path / "v.avi", frames=200, fps=25.0)

    class _Stationary(_StubDetector):
        def detect(self, image, provenance=None):
            side = 0.35 * max(image.shape[1], image.shape[0])
            return [
                _RawDetection(
                    box=BoundingBox(40.0, 40.0, 40.0 + side, 40.0 + side * 0.4),
                    label="animal",
                    confidence=0.9,
                )
            ]

    report = _check(
        video,
        profile,
        config,
        detector=_Stationary(lambda i: (1, 0.35)),
        window_seconds=4.0,
        window_count=1,
    )
    transit = next(c for c in report.checks if c.name == "transit")
    assert transit.verdict == FAIL
    assert "persisted without crossing" in transit.measured + transit.note


# -- 6. R6 gains the filename fallback ---------------------------------------


def test_a_time_in_the_filename_is_read_when_the_container_has_none(
    tmp_path, profile, config
) -> None:
    video = write_video(tmp_path / "lane_2024-03-05T13-04-28.avi", frames=30, fps=25.0)
    r6 = next(c for c in _check(video, profile, config).checks if c.requirement == "R6")
    read = next(p for p in r6.parts if p.attempted)
    assert read.passed
    assert "filename" in read.detail
    assert "2024-03-05" in r6.measured


def test_a_file_with_no_time_anywhere_fails_that_part(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "untimed.avi", frames=30, fps=25.0)
    r6 = next(c for c in _check(video, profile, config).checks if c.requirement == "R6")
    read = next(p for p in r6.parts if p.attempted)
    assert not read.passed
    assert r6.verdict == FAIL


# -- 7. the verdict names its configuration ----------------------------------


def test_the_report_names_what_it_judged_against(tmp_path, profile, config) -> None:
    video = write_video(tmp_path / "v.avi", frames=30, fps=25.0)
    report = _check(video, profile, config)
    assert report.profile_id.startswith(profile.skeleton.identifier)
    assert report.feature_set_version == profile.feature_set.version
    assert report.view == profile.skeleton.view
    described = report.describe()
    assert profile.skeleton.identifier in described
    assert profile.skeleton.view in described
