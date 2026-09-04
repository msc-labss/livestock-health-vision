"""Video ingest: registration, provenance, determinism, resume, timestamps, isolation."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lhv.errors import DerivedMediaIsolationError, MissingFieldError, RegistrationError
from lhv.ingest import Ingestor, register_source
from lhv.ingest.isolation import assert_isolated_output_root

from .conftest import FIXED_START, write_image_sequence, write_video

# -- 3.1 source registration and the provenance record ----------------------


def test_registered_source_carries_every_split_key(source) -> None:
    assert source.source_id and source.camera_id
    assert source.site_key and source.animal_set_key


@pytest.mark.parametrize("field_name", ["site_key", "camera_id", "source_id", "animal_set_key"])
def test_source_missing_a_required_key_is_refused_at_registration(
    field_name: str, video_path: Path
) -> None:
    kwargs = dict(
        source_id="s",
        camera_id="c",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(video_path),
        day_key="2024-03-01",
    )
    kwargs.pop(field_name)
    with pytest.raises(RegistrationError) as excinfo:
        register_source(**kwargs)
    assert excinfo.value.missing_field == field_name


@pytest.mark.parametrize("field_name", ["site_key", "camera_id"])
def test_source_with_an_empty_required_key_is_refused(field_name: str, video_path: Path) -> None:
    kwargs = dict(
        source_id="s",
        camera_id="c",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(video_path),
        day_key="2024-03-01",
    )
    kwargs[field_name] = "  "
    with pytest.raises(RegistrationError) as excinfo:
        register_source(**kwargs)
    assert excinfo.value.missing_field == field_name


def test_no_frames_are_emitted_from_an_unregistered_source(video_path: Path) -> None:
    """Registration is the gate: a refused source never reaches the decoder."""
    with pytest.raises(RegistrationError):
        register_source(
            source_id="s",
            camera_id="c",
            animal_set_key="herd-a",
            media_path=str(video_path),
            day_key="2024-03-01",
        )


def test_source_without_a_day_key_or_start_timestamp_is_refused(video_path: Path) -> None:
    with pytest.raises(RegistrationError) as excinfo:
        register_source(
            source_id="s",
            camera_id="c",
            site_key="site-a",
            animal_set_key="herd-a",
            media_path=str(video_path),
        )
    assert excinfo.value.missing_field == "day_key"


def test_frame_provenance_refuses_a_missing_field() -> None:
    from lhv.ingest import FrameProvenance

    with pytest.raises(MissingFieldError) as excinfo:
        FrameProvenance(
            source_id="s",
            camera_id="c",
            frame_index=0,
            animal_set_key="herd-a",
            day_key="2024-03-01",
        )
    assert excinfo.value.field_name == "site_key"


# -- 3.2 frame emission with full provenance --------------------------------


def test_every_emitted_frame_carries_complete_provenance(source, config) -> None:
    frames = list(Ingestor(source, config).iter_frames())
    assert frames
    for position, frame in enumerate(frames):
        provenance = frame.provenance
        assert provenance.source_id == source.source_id
        assert provenance.camera_id == source.camera_id
        assert provenance.site_key == source.site_key
        assert provenance.animal_set_key == source.animal_set_key
        assert provenance.day_key
        assert provenance.frame_index == position
        assert provenance.capture_timestamp is not None
        assert provenance.config_digest == config.digest
        assert frame.image is not None


def test_provenance_exposes_the_split_keys_the_harness_needs(source, config) -> None:
    frame = next(iter(Ingestor(source, config).iter_frames()))
    keys = frame.provenance.split_keys()
    assert set(keys) >= {"animal_set", "day", "site", "camera", "source"}
    assert all(keys.values())


def test_day_key_follows_the_capture_timestamp_when_it_is_reliable(source, config) -> None:
    frame = next(iter(Ingestor(source, config).iter_frames()))
    assert frame.provenance.day_key == FIXED_START.date().isoformat()


def test_image_sequence_source_emits_provenance_too(tmp_path: Path, config) -> None:
    directory = write_image_sequence(tmp_path / "seq", frames=6)
    source = register_source(
        source_id="synthetic/seq",
        camera_id="cam-2",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(directory),
        day_key="2024-03-02",
        kind="image_sequence",
    )
    frames = list(Ingestor(source, config).iter_frames())
    assert len(frames) == 6
    assert all(f.provenance.day_key == "2024-03-02" for f in frames)


# -- 3.3 deterministic iteration --------------------------------------------


def test_two_runs_over_the_same_source_agree(source, config) -> None:
    first = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    second = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    assert len(first) == len(second)
    assert [p.frame_index for p in first] == [p.frame_index for p in second]
    assert first == second


def test_stride_is_part_of_the_configuration_not_of_the_run(source, config) -> None:
    import dataclasses

    strided = dataclasses.replace(config, ingest=dataclasses.replace(config.ingest, frame_stride=3))
    indices = [f.index for f in Ingestor(source, strided).iter_frames(decode=False)]
    assert indices == sorted(indices)
    assert all(index % 3 == 0 for index in indices)
    assert strided.digest != config.digest


# -- 3.4 resumable iteration ------------------------------------------------


def test_resume_continues_at_the_next_frame_without_repeating(source, config, output_root) -> None:
    ingestor = Ingestor(source, config, output_root=output_root)

    interrupted = []
    for frame in ingestor.iter_frames(decode=False):
        interrupted.append(frame.index)
        if len(interrupted) == 7:
            ingestor.record_position(frame.index, emitted=len(interrupted))
            break

    checkpoint = Ingestor(source, config, output_root=output_root).recorded_position()
    assert checkpoint is not None
    assert checkpoint.last_emitted_index == interrupted[-1]

    resumed = [
        frame.index
        for frame in Ingestor(source, config, output_root=output_root).iter_frames(
            resume_from=checkpoint.last_emitted_index, decode=False
        )
    ]
    assert resumed[0] == interrupted[-1] + 1
    assert not set(resumed) & set(interrupted)

    complete = [f.index for f in Ingestor(source, config).iter_frames(decode=False)]
    assert interrupted + resumed == complete


def test_resume_keeps_the_stride_phase_of_an_uninterrupted_run(source, config, output_root) -> None:
    import dataclasses

    strided = dataclasses.replace(config, ingest=dataclasses.replace(config.ingest, frame_stride=4))
    complete = [f.index for f in Ingestor(source, strided).iter_frames(decode=False)]
    cut = complete[2]
    resumed = [
        f.index
        for f in Ingestor(source, strided, output_root=output_root).iter_frames(
            resume_from=cut, decode=False
        )
    ]
    assert resumed == complete[3:]


# -- 3.5 timestamp integrity ------------------------------------------------


def test_missing_timestamp_is_flagged_rather_than_defaulted(tmp_path: Path, config) -> None:
    """A source with no start timestamp cannot produce one; it must say so."""
    directory = write_image_sequence(tmp_path / "seq", frames=4)
    source = register_source(
        source_id="synthetic/no-clock",
        camera_id="cam-3",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(directory),
        day_key="2024-03-03",
        kind="image_sequence",
    )
    ingestor = Ingestor(source, config)
    frames = list(ingestor.iter_frames(decode=False))
    assert frames
    for frame in frames:
        assert frame.provenance.capture_timestamp is None
        assert frame.provenance.timestamp_reliable is False
    assert ingestor.report.unreliable_timestamp_indices == [f.index for f in frames]
    # The day key still exists, taken from registration rather than invented.
    assert all(f.provenance.day_key == "2024-03-03" for f in frames)


def test_downstream_can_filter_on_the_unreliable_mark(tmp_path: Path, config) -> None:
    directory = write_image_sequence(tmp_path / "seq", frames=4)
    source = register_source(
        source_id="s",
        camera_id="c",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(directory),
        day_key="2024-03-03",
        kind="image_sequence",
    )
    frames = list(Ingestor(source, config).iter_frames(decode=False))
    assert [f for f in frames if f.provenance.timestamp_reliable] == []


def test_exclude_policy_drops_unreliable_frames_and_counts_them(tmp_path: Path, config) -> None:
    import dataclasses

    directory = write_image_sequence(tmp_path / "seq", frames=4)
    source = register_source(
        source_id="s",
        camera_id="c",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(directory),
        day_key="2024-03-03",
        kind="image_sequence",
    )
    strict = dataclasses.replace(
        config, ingest=dataclasses.replace(config.ingest, unreliable_timestamp_policy="exclude")
    )
    ingestor = Ingestor(source, strict)
    assert list(ingestor.iter_frames(decode=False)) == []
    assert ingestor.report.frames_excluded == 4


def test_non_monotonic_timestamps_are_reported(tmp_path: Path, config, monkeypatch) -> None:
    from datetime import timedelta

    from lhv.ingest import stream as stream_module

    path = write_video(tmp_path / "media" / "lane.avi", frames=6)
    source = register_source(
        source_id="synthetic/skewed",
        camera_id="cam-4",
        site_key="site-a",
        animal_set_key="herd-a",
        media_path=str(path),
        start_timestamp=FIXED_START,
    )
    ingestor = stream_module.Ingestor(source, config)

    # Frame 3 claims a capture time earlier than frame 2's.
    skewed = {0: 0.0, 1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0, 5: 5.0}
    calls = {"n": -1}

    def fake_timestamp(offset_seconds):
        calls["n"] += 1
        seconds = skewed.get(calls["n"], calls["n"])
        return FIXED_START + timedelta(seconds=seconds), True

    monkeypatch.setattr(ingestor, "_timestamp_for", fake_timestamp)
    frames = list(ingestor.iter_frames(decode=False))
    assert len(frames) == 6

    anomalies = ingestor.report.timestamp_anomalies
    assert [a.frame_index for a in anomalies] == [3]
    assert anomalies[0].kind == "non_monotonic"
    assert "non-monotonic timestamp at frame 3" in ingestor.report.describe()


# -- 3.6 derived-media isolation --------------------------------------------


def test_ingest_refuses_a_version_controlled_output_root(source, config, tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    (repository / "artifacts").mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)

    with pytest.raises(DerivedMediaIsolationError) as excinfo:
        Ingestor(source, config, output_root=repository / "artifacts")
    assert str(repository / "artifacts") in excinfo.value.path


def test_ingest_fails_before_writing_anything(source, config, tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    target = repository / "artifacts"
    target.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(repository)], check=True)

    with pytest.raises(DerivedMediaIsolationError):
        Ingestor(source, config, output_root=target)
    assert list(target.iterdir()) == [], "nothing may be written before the check passes"


def test_an_ignored_directory_inside_a_repository_is_accepted(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text("/derived/\n", encoding="utf-8")
    target = repository / "derived"
    target.mkdir()
    assert assert_isolated_output_root(target) == target.resolve()


def test_a_directory_outside_any_repository_is_accepted(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    assert assert_isolated_output_root(outside) == outside.resolve()


def test_this_repository_refuses_its_own_source_tree() -> None:
    """The guard must hold for the actual repository, not only a synthetic one."""
    with pytest.raises(DerivedMediaIsolationError):
        assert_isolated_output_root(Path(__file__).resolve().parent.parent / "src")
