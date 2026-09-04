"""Declarative source layouts: matching, grouping, and refusing what is not declared."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lhv.datasets import AccessTerms, DatasetRegistration, load_registration
from lhv.datasets.layout import LayoutSpec, layout_from_dict, sources_from_layout
from lhv.errors import RegistrationError

TRACKLETS = LayoutSpec(
    path="MultiCamCows2024Root/{day}/{animal}/{frame}_{camera}.jpg",
    kind="image_sequence",
    group_by=("day", "animal", "camera"),
    order_by="frame",
    identity_field="animal",
    camera_map={"1": "camera1", "2": "camera2", "3": "camera3"},
)

FOOTAGE = LayoutSpec(
    path="MultiCamCows2024Root/videos/{camera}/{day}/{timestamp}.mp4",
    kind="video",
    group_by=("camera", "day", "timestamp"),
    timestamp_field="timestamp",
    timestamp_format="%Y-%m-%dT%H-%M-%S",
)


@pytest.fixture
def registration():
    return DatasetRegistration(
        name="mcc",
        version="1",
        site_key="site-a",
        camera_ids=("camera1", "camera2", "camera3"),
        animal_set_keys=("herd",),
        access=AccessTerms(licence="test"),
    )


def _tree(root: Path) -> Path:
    """A miniature of the real release, with its exact conventions."""
    for day in ("2023Aug14", "2023Aug15"):
        for animal in ("001", "056"):
            directory = root / "MultiCamCows2024Root" / day / animal
            directory.mkdir(parents=True, exist_ok=True)
            for frame in range(4):
                for camera in ("1", "2", "3"):
                    (directory / f"{frame:05d}_{camera}.jpg").write_bytes(b"x")
    for camera in ("camera1", "camera2"):
        directory = root / "MultiCamCows2024Root" / "videos" / camera / "2023Aug14"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "2023-08-14T13-04-28.mp4").write_bytes(b"x")
    return root


# -- matching ----------------------------------------------------------------


def test_the_template_matches_a_real_tracklet_path() -> None:
    captures = TRACKLETS.match("MultiCamCows2024Root/2023Aug16/056/00026_2.jpg")
    assert captures == {"day": "2023Aug16", "animal": "056", "frame": "00026", "camera": "2"}
    assert TRACKLETS.camera_id(captures) == "camera2"
    assert TRACKLETS.day_key(captures) == "2023Aug16"
    assert TRACKLETS.identity(captures) == "056"


def test_the_template_matches_a_real_footage_path() -> None:
    captures = FOOTAGE.match(
        "MultiCamCows2024Root/videos/camera2/2023Aug19/2023-08-19T13-04-28.mp4"
    )
    assert captures["camera"] == "camera2"
    assert FOOTAGE.timestamp(captures) == datetime(2023, 8, 19, 13, 4, 28, tzinfo=UTC)


def test_a_path_that_does_not_fit_the_template_is_not_matched() -> None:
    assert TRACKLETS.match("MultiCamCows2024Root/2023Aug16/056/notes.txt") is None
    assert TRACKLETS.match("somewhere/else/00026_2.jpg") is None


def test_an_unparseable_timestamp_yields_none_rather_than_a_default() -> None:
    assert FOOTAGE.timestamp({"timestamp": "not-a-time"}) is None
    assert TRACKLETS.timestamp({"frame": "00026"}) is None


def test_the_glob_narrows_the_search_without_matching_everything() -> None:
    assert TRACKLETS.glob == "MultiCamCows2024Root/*/*/*_*.jpg"
    assert FOOTAGE.glob == "MultiCamCows2024Root/videos/*/*/*.mp4"


# -- validation --------------------------------------------------------------


def test_a_layout_naming_a_field_it_does_not_capture_is_refused() -> None:
    with pytest.raises(RegistrationError, match="identity_field"):
        layout_from_dict({"path": "{day}/{frame}.jpg", "identity_field": "animal"}, where="<test>")


def test_a_single_camera_layout_need_not_capture_a_camera() -> None:
    """Not every source has more than one camera to distinguish."""
    spec = layout_from_dict({"path": "{day}/{frame}.jpg", "order_by": "frame"}, where="<test>")
    assert spec.camera_id({"day": "d", "frame": "1"}) == ""


def test_an_unknown_layout_key_is_refused() -> None:
    with pytest.raises(RegistrationError, match="unknown layout keys"):
        layout_from_dict({"path": "{day}/{frame}.jpg", "colour": "blue"}, where="<test>")


def test_an_unknown_layout_kind_is_refused() -> None:
    with pytest.raises(RegistrationError, match="unknown layout kind"):
        layout_from_dict({"path": "{day}/{frame}.jpg", "kind": "hologram"}, where="<test>")


def test_asking_for_an_undeclared_layout_names_what_exists(registration) -> None:
    with pytest.raises(RegistrationError, match="no layout named"):
        registration.layout("nope")


# -- grouping into sources ---------------------------------------------------


def test_one_source_per_animal_day_and_camera(tmp_path, registration) -> None:
    sources = sources_from_layout(registration, TRACKLETS, _tree(tmp_path))
    # 2 days x 2 animals x 3 cameras
    assert len(sources) == 12
    assert {s.camera_id for s in sources} == {"camera1", "camera2", "camera3"}
    assert {s.animal_id for s in sources} == {"001", "056"}
    assert {s.day_key for s in sources} == {"2023Aug14", "2023Aug15"}


def test_a_source_carries_exactly_its_own_camera_frames(tmp_path, registration) -> None:
    """One directory interleaves three cameras, so a directory is not a source."""
    sources = sources_from_layout(registration, TRACKLETS, _tree(tmp_path))
    one = next(s for s in sources if s.camera_id == "camera2" and s.animal_id == "056")
    assert len(one.media_paths) == 4
    assert all(p.endswith("_2.jpg") for p in one.media_paths)
    assert one.media_paths == tuple(sorted(one.media_paths))


def test_split_keys_come_from_the_registration_not_the_layout(tmp_path, registration) -> None:
    for source in sources_from_layout(registration, TRACKLETS, _tree(tmp_path)):
        assert source.site_key == registration.site_key
        assert source.animal_set_key == registration.animal_set_keys[0]


def test_video_sources_are_one_file_each_with_a_real_capture_time(tmp_path, registration) -> None:
    sources = sources_from_layout(registration, FOOTAGE, _tree(tmp_path))
    assert len(sources) == 2
    for source in sources:
        assert source.kind == "video"
        assert source.media_paths == ()
        assert source.start_timestamp == datetime(2023, 8, 14, 13, 4, 28, tzinfo=UTC)


def test_a_camera_the_registration_never_declared_is_refused(tmp_path) -> None:
    """The registration is the authority on which cameras exist."""
    narrow = DatasetRegistration(
        name="mcc",
        version="1",
        site_key="site-a",
        camera_ids=("camera1",),
        animal_set_keys=("herd",),
        access=AccessTerms(licence="test"),
    )
    with pytest.raises(RegistrationError) as excinfo:
        sources_from_layout(narrow, TRACKLETS, _tree(tmp_path))
    assert excinfo.value.missing_field == "camera_ids"
    assert "camera2" in str(excinfo.value) or "camera3" in str(excinfo.value)


def test_tracklet_sources_have_no_capture_time_and_say_so(tmp_path, registration) -> None:
    """Pre-cropped stills carry no clock; ingest must not invent one."""
    from lhv.config import ResolvedConfig
    from lhv.ingest import Ingestor

    sources = sources_from_layout(registration, TRACKLETS, _tree(tmp_path))
    source = sources[0]
    assert source.start_timestamp is None
    assert source.day_key == "2023Aug14"

    config = ResolvedConfig(
        species_profile="cattle",
        species_profile_version="0",
        dataset_name="mcc",
        dataset_version="1",
    )
    ingestor = Ingestor(source, config)
    frames = list(ingestor.iter_frames(decode=False))
    assert len(frames) == 4
    assert all(not f.provenance.timestamp_reliable for f in frames)
    assert all(f.provenance.capture_timestamp is None for f in frames)
    assert all(f.provenance.day_key == "2023Aug14" for f in frames)


# -- the shipped registration -------------------------------------------------


def test_the_shipped_registration_declares_both_components() -> None:
    registration = load_registration("multicamcows2024")
    assert set(registration.layouts) == {"tracklets", "footage"}
    registration.layout("tracklets").validate()
    registration.layout("footage").validate()


def test_the_shipped_layout_matches_the_paths_the_archive_actually_contains() -> None:
    """Read from the archive's central directory before it was downloaded."""
    registration = load_registration("multicamcows2024")
    tracklets = registration.layout("tracklets")
    footage = registration.layout("footage")

    captures = tracklets.match("MultiCamCows2024Root/2023Aug16/056/00026_2.jpg")
    assert captures is not None
    assert tracklets.camera_id(captures) in registration.camera_ids
    assert tracklets.identity(captures) == "056"

    captures = footage.match(
        "MultiCamCows2024Root/videos/camera2/2023Aug19/2023-08-19T13-04-28.mp4"
    )
    assert captures is not None
    assert footage.camera_id(captures) in registration.camera_ids
    assert footage.timestamp(captures) is not None
