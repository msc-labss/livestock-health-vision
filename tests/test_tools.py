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
