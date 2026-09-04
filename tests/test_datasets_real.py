"""Checks that run against the real public datasets, when they are on disk.

Marked ``dataset`` and skipped otherwise, so continuous integration stays
runnable without 40 GB of livestock footage. These are the checks that turn the
registrations' declared figures into verified ones.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lhv.config import ResolvedConfig
from lhv.datasets import load_registration
from lhv.ingest import Ingestor

pytestmark = pytest.mark.dataset

DATA = Path(__file__).resolve().parent.parent / "data"
CEV = DATA / "cattleeyeview"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not present at {path}")
    return path


@pytest.fixture
def cattleeyeview():
    _require(CEV / "videos", "CattleEyeView videos")
    return load_registration("cattleeyeview")


@pytest.fixture
def config(cattleeyeview):
    return ResolvedConfig(
        species_profile="cattle",
        species_profile_version="0",
        dataset_name=cattleeyeview.name,
        dataset_version=cattleeyeview.version,
    )


# -- content counts, checked against the download rather than the paper -------


def test_the_sequence_count_matches_the_paper(cattleeyeview) -> None:
    report = cattleeyeview.verify(CEV)
    sequences = next(c for c in report.counts if c.name == "sequences")
    assert sequences.observed == 14
    assert sequences.matches


def test_the_extracted_frame_count_matches_the_paper(cattleeyeview) -> None:
    """Counted inside images.tar.gz without unpacking its 12 GB."""
    if not (CEV / "images.tar.gz").exists():
        pytest.skip("images.tar.gz not present")
    report = cattleeyeview.verify(CEV)
    frames = next(c for c in report.counts if c.name == "frames")
    assert frames.declared == 30703
    assert frames.observed == 30703
    assert frames.matches


def test_the_videos_hold_three_frames_the_extracted_set_does_not(cattleeyeview) -> None:
    """Both figures are right and measure different things.

    The paper's 30,703 is the extracted set. The videos decode to 30,706,
    because the final frame of sequences 03, 05 and 06 was never extracted.
    """
    report = cattleeyeview.verify(CEV)
    in_videos = next(c for c in report.counts if c.name == "frames_in_videos")
    assert in_videos.observed == 30706
    assert in_videos.matches, "the measured figure is recorded, so this is a regression check"

    if not (CEV / "images.tar.gz").exists():
        pytest.skip("images.tar.gz not present")
    extracted = next(c for c in report.counts if c.name == "frames")
    assert in_videos.observed - extracted.observed == 3


def test_the_release_image_paths_resolve_to_video_frames(cattleeyeview) -> None:
    """images/<sequence>.mp4/<frame>.jpg, one-based against a zero-based video."""
    layout = cattleeyeview.layout("frames")
    captures = layout.match("images/03.mp4/00871.jpg")
    assert captures == {"sequence": "03", "frame": "00871"}
    assert int(captures["frame"]) - 1 == 870


# -- the recovered recording times -------------------------------------------


def test_every_sequence_has_a_recording_time(cattleeyeview) -> None:
    sources = cattleeyeview.sources("footage", CEV)
    assert len(sources) == 14
    assert all(s.start_timestamp is not None for s in sources)


def test_the_recording_window_matches_what_the_paper_states(cattleeyeview) -> None:
    """The paper says 2021-11-09 to 2022-03-04; the burned-in overlays agree."""
    sources = sorted(cattleeyeview.sources("footage", CEV), key=lambda s: s.source_id)
    assert sources[0].start_timestamp.date() == date(2021, 11, 9)
    assert sources[-1].start_timestamp.date() == date(2022, 3, 4)


def test_recording_times_increase_with_the_sequence_number(cattleeyeview) -> None:
    sources = sorted(cattleeyeview.sources("footage", CEV), key=lambda s: s.source_id)
    stamps = [s.start_timestamp for s in sources]
    assert stamps == sorted(stamps)


def test_the_sequences_span_thirteen_distinct_days(cattleeyeview) -> None:
    """Two sequences share 2021-12-19, so a day split is not a sequence split."""
    sources = cattleeyeview.sources("footage", CEV)
    days = {s.day_key for s in sources}
    assert len(days) == 13
    assert len(sources) == 14


# -- ingest over real media ---------------------------------------------------


def test_ingest_emits_every_frame_of_a_real_sequence(cattleeyeview, config) -> None:
    source = next(s for s in cattleeyeview.sources("footage", CEV) if s.source_id.endswith("/01"))
    frames = list(Ingestor(source, config).iter_frames(decode=False))
    assert len(frames) == 1226
    assert [f.index for f in frames] == list(range(1226))


def test_real_frames_carry_reliable_capture_times(cattleeyeview, config) -> None:
    source = next(s for s in cattleeyeview.sources("footage", CEV) if s.source_id.endswith("/01"))
    frames = list(Ingestor(source, config).iter_frames(decode=False))

    assert all(f.provenance.timestamp_reliable for f in frames)
    assert frames[0].provenance.capture_timestamp == source.start_timestamp
    # 8 fps: an eighth of a second per frame.
    step = (
        frames[1].provenance.capture_timestamp - frames[0].provenance.capture_timestamp
    ).total_seconds()
    assert step == pytest.approx(0.125)
    assert all(f.provenance.day_key == "2021-11-09" for f in frames)


def test_real_frames_carry_the_registration_split_keys(cattleeyeview, config) -> None:
    source = cattleeyeview.sources("footage", CEV)[0]
    frame = next(iter(Ingestor(source, config).iter_frames(decode=False)))
    keys = frame.provenance.split_keys()
    assert keys["site"] == cattleeyeview.site_key
    assert keys["camera"] in cattleeyeview.camera_ids
    assert keys["day"] == "2021-11-09"


def test_ingest_over_a_real_sequence_is_deterministic(cattleeyeview, config) -> None:
    source = cattleeyeview.sources("footage", CEV)[0]
    first = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    second = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    assert first == second


# -- the sampling rates that limit what can be measured -----------------------


def test_the_sequences_are_1920x1080_at_three_declared_frame_rates(cattleeyeview) -> None:
    """The paper states neither. Both bound what the phenotype layer can report."""
    import cv2

    rates, sizes = set(), set()
    for source in cattleeyeview.sources("footage", CEV):
        capture = cv2.VideoCapture(source.media_path)
        try:
            rates.add(round(capture.get(cv2.CAP_PROP_FPS)))
            sizes.add(
                (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            )
        finally:
            capture.release()

    assert sizes == {(1920, 1080)}
    assert rates == {3, 5, 8}


def test_the_slow_sequences_cannot_resolve_a_stride(cattleeyeview) -> None:
    """Five of fourteen sequences run at 3 fps, below the profile's requirement."""
    import cv2

    from lhv.profiles import load_profile

    required = load_profile("cattle").feature_set.get("stride_frequency_front")
    assert required.requires_sampling_hz == 5.0

    too_slow = 0
    for source in cattleeyeview.sources("footage", CEV):
        capture = cv2.VideoCapture(source.media_path)
        try:
            if capture.get(cv2.CAP_PROP_FPS) < required.requires_sampling_hz:
                too_slow += 1
        finally:
            capture.release()
    assert too_slow == 5
