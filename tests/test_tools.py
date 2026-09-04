"""The operational tools: segment planning, and the archive component selector."""

from __future__ import annotations

import pytest
from tools.fetch_archive import plan_segments
from tools.prepare_multicamcows import _wanted

TOTAL = 39_270_266_920


def test_segments_cover_the_whole_file_exactly() -> None:
    segments = plan_segments(TOTAL, 8)
    assert segments[0]["start"] == 0
    assert segments[-1]["end"] == TOTAL - 1
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert later["start"] == earlier["end"] + 1
    assert sum(s["end"] - s["start"] + 1 for s in segments) == TOTAL


def test_a_fresh_plan_has_nothing_done() -> None:
    assert all(s["done"] == 0 for s in plan_segments(TOTAL, 8))


def test_an_existing_prefix_is_credited_to_the_segments_it_covers() -> None:
    """An interrupted single-connection download must not be re-fetched."""
    segments = plan_segments(TOTAL, 8)
    length = segments[0]["end"] - segments[0]["start"] + 1
    have = length + 1000

    resumed = plan_segments(TOTAL, 8, have=have)
    assert resumed[0]["done"] == length, "the first segment is complete"
    assert resumed[1]["done"] == 1000, "the remainder credits the second"
    assert all(s["done"] == 0 for s in resumed[2:])
    assert sum(s["done"] for s in resumed) == have


def test_a_prefix_longer_than_the_file_is_clamped() -> None:
    segments = plan_segments(1000, 4, have=99_999)
    assert sum(s["done"] for s in segments) == 1000
    for segment in segments:
        assert segment["done"] <= segment["end"] - segment["start"] + 1


@pytest.mark.parametrize("connections", [1, 2, 3, 8, 16])
def test_any_connection_count_still_covers_the_file(connections: int) -> None:
    segments = plan_segments(TOTAL, connections)
    assert sum(s["end"] - s["start"] + 1 for s in segments) == TOTAL
    assert segments[-1]["end"] == TOTAL - 1


# -- component selection ----------------------------------------------------

IMAGE = "2inu67jru7a6821kkgehxg3cv2/MultiCamCows2024Root/2023Aug16/004/00023_2.jpg"
VIDEO = "2inu67jru7a6821kkgehxg3cv2/MultiCamCows2024Root/videos/2023Aug16/cam1.mp4"
README = "2inu67jru7a6821kkgehxg3cv2/readme.txt"


def test_the_images_component_excludes_the_monitor_footage() -> None:
    """33 of the archive's 36.6 GB is footage; the identity labels are in the 3.4 GB."""
    assert _wanted(IMAGE, "images")
    assert _wanted(README, "images")
    assert not _wanted(VIDEO, "images")


def test_the_videos_component_is_only_the_footage() -> None:
    assert _wanted(VIDEO, "videos")
    assert not _wanted(IMAGE, "videos")
    assert not _wanted(README, "videos")


def test_all_selects_everything() -> None:
    assert all(_wanted(name, "all") for name in (IMAGE, VIDEO, README))


def test_a_full_length_file_without_a_progress_record_is_not_trusted(tmp_path) -> None:
    """A segmented run pre-allocates the file, so its size proves nothing.

    Crediting a pre-allocated file as complete would declare an archive of zeros
    finished and hand it to the unpacker.
    """
    from tools.fetch_archive import Segmented

    target = tmp_path / "archive.zip"
    with target.open("wb") as handle:
        handle.truncate(TOTAL)

    planner = Segmented.__new__(Segmented)
    planner.target = target
    planner.state_path = target.with_suffix(target.suffix + ".progress.json")
    planner.total = TOTAL
    planner.connections = 8

    segments = planner._plan()
    assert sum(s["done"] for s in segments) == 0, "nothing may be assumed complete"


def test_a_short_file_is_still_credited_as_a_prefix(tmp_path) -> None:
    from tools.fetch_archive import Segmented

    target = tmp_path / "archive.zip"
    with target.open("wb") as handle:
        handle.truncate(1_000_000)

    planner = Segmented.__new__(Segmented)
    planner.target = target
    planner.state_path = target.with_suffix(target.suffix + ".progress.json")
    planner.total = TOTAL
    planner.connections = 8

    assert sum(s["done"] for s in planner._plan()) == 1_000_000


def test_a_matching_progress_record_is_used_verbatim(tmp_path) -> None:
    import json

    from tools.fetch_archive import Segmented, plan_segments

    target = tmp_path / "archive.zip"
    with target.open("wb") as handle:
        handle.truncate(TOTAL)

    recorded = plan_segments(TOTAL, 8)
    recorded[3]["done"] = 12345
    state = target.with_suffix(target.suffix + ".progress.json")
    state.write_text(json.dumps({"total": TOTAL, "segments": recorded}))

    planner = Segmented.__new__(Segmented)
    planner.target = target
    planner.state_path = state
    planner.total = TOTAL
    planner.connections = 8

    assert planner._plan()[3]["done"] == 12345


# -- Google Drive fails with a 200 and an HTML body --------------------------


def test_the_quota_page_is_recognised_and_named() -> None:
    """Drive answers 200 with this when a shared folder passes its daily quota."""
    from tools.fetch_drive import looks_like_drive_error

    page = (
        b"<!DOCTYPE html><html><head><title>Google Drive - Quota exceeded</title>"
        b"</head><body>Sorry, you can't view or download this file at this time.</body></html>"
    )
    problem = looks_like_drive_error(page)
    assert "Quota exceeded" in problem


def test_a_sign_in_page_is_recognised() -> None:
    from tools.fetch_drive import looks_like_drive_error

    page = b"<!DOCTYPE html><html><head><title>Sign in - Google Accounts</title></head></html>"
    assert "not publicly readable" in looks_like_drive_error(page)


@pytest.mark.parametrize(
    "payload",
    [
        b"\x00\x00\x00\x20ftypisom",  # mp4
        b"\x1f\x8b\x08\x00",  # gzip
        b'{"images": [], "annotations": []}',  # json
        b"\xff\xd8\xff\xe0\x00\x10JFIF",  # jpeg
    ],
)
def test_a_real_file_is_not_mistaken_for_an_error_page(payload: bytes) -> None:
    from tools.fetch_drive import looks_like_drive_error

    assert looks_like_drive_error(payload) == ""


def test_a_refused_download_writes_nothing(tmp_path, monkeypatch) -> None:
    """The whole point: an interstitial must never land on disk as the file."""
    from tools import fetch_drive

    monkeypatch.setattr(
        fetch_drive,
        "_get",
        lambda url, timeout=600: b"<!DOCTYPE html><title>Google Drive - Quota exceeded</title>",
    )
    entry = fetch_drive.Entry(id="x", name="videos/01.mp4", mime="video/mp4", size=0)
    target = tmp_path / "01.mp4"

    with pytest.raises(fetch_drive.QuotaExceeded, match="Quota exceeded"):
        fetch_drive.download(entry, target)
    assert not target.exists()


def test_a_real_download_is_written(tmp_path, monkeypatch) -> None:
    from tools import fetch_drive

    monkeypatch.setattr(fetch_drive, "_get", lambda url, timeout=600: b"\x1f\x8b\x08\x00payload")
    entry = fetch_drive.Entry(id="x", name="images.tar.gz", mime="application/gzip", size=9)
    target = tmp_path / "images.tar.gz"

    assert fetch_drive.download(entry, target) == target
    assert target.read_bytes().startswith(b"\x1f\x8b")
