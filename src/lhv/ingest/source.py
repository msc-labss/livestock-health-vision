"""Source registration.

A source is refused unless it carries the keys every later stage depends on. The
refusal happens at registration and names the missing field, because a source
that reaches the decoder without a site key has already cost the expensive part
of the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from ..errors import RegistrationError
from ..schema import Record, opt, req
from .provenance import SourceProvenance

if TYPE_CHECKING:  # ingest sits below datasets; the dependency is for types only
    from ..datasets.registration import DatasetRegistration

__all__ = ["RegisteredSource", "register_source", "register_sources_from_dataset"]

_REQUIRED_KEYS = ("source_id", "camera_id", "site_key", "animal_set_key")


@dataclass(frozen=True)
class RegisteredSource(Record):
    """One piece of source material, with the keys that make it ingestible."""

    SCHEMA_NAME = "registered_source"
    # 2: added media_paths, for sources whose frames are an explicit ordered
    #    list rather than everything in a directory.
    # 3: added view, the camera geometry the source was recorded under. Carried
    #    with the source rather than in configuration, because it is a property
    #    of the recording and belongs with the recording's provenance.
    SCHEMA_VERSION = "3"

    source_id: str = req()
    camera_id: str = req()
    site_key: str = req()
    animal_set_key: str = req()
    media_path: str = req()
    dataset_name: str = opt("")
    dataset_version: str = opt("")
    # Either a declared day key or a start timestamp must be present, so that a
    # day key can be attached to every frame without inventing one.
    day_key: str = opt("")
    start_timestamp: datetime | None = opt(None)
    kind: str = opt("video")  # "video" or "image_sequence"
    # When present, exactly these files in exactly this order are the source's
    # frames. Used where one directory interleaves several cameras.
    media_paths: tuple[str, ...] = opt(())
    animal_id: str = opt("")  # ground-truth identity, where the source carries one
    # Camera geometry, in the vocabulary a skeleton's view uses. Empty means
    # undeclared, which pose estimation refuses rather than assumes.
    view: str = opt("")
    notes: str = opt("")

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in _REQUIRED_KEYS:
            if not str(getattr(self, name)).strip():
                raise RegistrationError(
                    f"source registration: {name!r} is empty; ingest refuses a source that "
                    f"cannot supply it",
                    missing_field=name,
                )
        if not self.day_key and self.start_timestamp is None:
            raise RegistrationError(
                f"source {self.source_id!r}: neither 'day_key' nor 'start_timestamp' was "
                f"supplied, so no day key can be attached to its frames without inventing one",
                missing_field="day_key",
            )
        if self.kind not in {"video", "image_sequence"}:
            raise RegistrationError(
                f"source {self.source_id!r}: unknown kind {self.kind!r}", missing_field="kind"
            )

    @property
    def default_day_key(self) -> str:
        if self.day_key:
            return self.day_key
        assert self.start_timestamp is not None  # guaranteed by __post_init__
        return self.start_timestamp.date().isoformat()

    def provenance(self, *, frame_rate: float = 0.0, frame_count: int = 0) -> SourceProvenance:
        return SourceProvenance(
            source_id=self.source_id,
            camera_id=self.camera_id,
            site_key=self.site_key,
            animal_set_key=self.animal_set_key,
            dataset_name=self.dataset_name or "unregistered",
            dataset_version=self.dataset_version or "0",
            media_path=self.media_path,
            frame_rate=frame_rate,
            frame_count=frame_count,
        )


def register_source(**kwargs) -> RegisteredSource:
    """Register a source, refusing by name anything a later stage would need."""
    missing = [key for key in _REQUIRED_KEYS if key not in kwargs or kwargs[key] is None]
    if missing:
        raise RegistrationError(
            f"source registration: missing required field {missing[0]!r}",
            missing_field=missing[0],
        )
    if "media_path" not in kwargs:
        raise RegistrationError(
            "source registration: missing required field 'media_path'",
            missing_field="media_path",
        )
    return RegisteredSource(**kwargs)


def register_sources_from_dataset(
    registration: DatasetRegistration,
    *,
    data_root: str | Path | None = None,
    camera_for: callable | None = None,
) -> tuple[RegisteredSource, ...]:
    """Enumerate the media beneath a registered dataset as registered sources.

    Split keys come from the dataset registration, so a source cannot be
    ingested under keys the dataset never declared.
    """
    root = registration.local_root(data_root)
    if not root.exists():
        raise RegistrationError(
            f"dataset {registration.name!r} is registered but not present at {root}"
        )

    media = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in registration.media_extensions
    )
    videos = [p for p in media if p.suffix.lower() in {".mp4", ".avi", ".mkv", ".mov"}]

    sources: list[RegisteredSource] = []
    if videos:
        for path in videos:
            relative = path.relative_to(root)
            sources.append(
                register_source(
                    source_id=f"{registration.name}/{relative.as_posix()}",
                    camera_id=(camera_for(relative) if camera_for else registration.camera_ids[0]),
                    site_key=registration.site_key,
                    animal_set_key=registration.animal_set_keys[0],
                    media_path=str(path),
                    dataset_name=registration.name,
                    dataset_version=registration.version,
                    day_key=relative.parts[0] if len(relative.parts) > 1 else registration.version,
                    kind="video",
                    view=registration.view,
                )
            )
        return tuple(sources)

    # No video: treat each leaf directory of images as one image-sequence source.
    directories = sorted({p.parent for p in media})
    for directory in directories:
        relative = directory.relative_to(root)
        sources.append(
            register_source(
                source_id=f"{registration.name}/{relative.as_posix() or '.'}",
                camera_id=camera_for(relative) if camera_for else registration.camera_ids[0],
                site_key=registration.site_key,
                animal_set_key=registration.animal_set_keys[0],
                media_path=str(directory),
                dataset_name=registration.name,
                dataset_version=registration.version,
                day_key=relative.parts[0] if relative.parts else registration.version,
                kind="image_sequence",
                view=registration.view,
            )
        )
    return tuple(sources)
