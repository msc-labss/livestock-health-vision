"""Declarative source layouts.

A registered dataset says how its files are arranged, and this module turns that
statement into registered sources. The alternative — a hand-written adapter per
dataset — puts each source's directory conventions into Python, which is exactly
the thing the proposal says should stay configuration so that adding a source
does not mean adding code.

A layout names a path template with `{field}` captures, which fields identify
one source, which capture carries the ground-truth animal identity, and how a
capture becomes a timestamp. Everything downstream then reads ordinary
provenance and never learns which dataset it came from.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..errors import RegistrationError

__all__ = ["LayoutSpec", "layout_from_dict", "relative_paths", "sources_from_layout"]

_FIELD = re.compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class LayoutSpec:
    """How one dataset's files are arranged beneath its root."""

    # e.g. "MultiCamCows2024Root/{day}/{animal}/{frame}_{camera}.jpg"
    path: str
    # "image_sequence" groups many files into one source; "video" is one file each.
    kind: str = "image_sequence"
    # Which captures together identify one source.
    group_by: tuple[str, ...] = ()
    # Which capture orders frames within a source.
    order_by: str = ""
    # Which capture carries the camera, the day, and the ground-truth identity.
    camera_field: str = "camera"
    day_field: str = "day"
    identity_field: str = ""
    # Raw capture -> the camera identifier the registration declares.
    camera_map: dict[str, str] = field(default_factory=dict)
    # Which capture carries a capture time, and how to read it.
    timestamp_field: str = ""
    timestamp_format: str = ""
    # Capture times a release does not put in its paths, keyed by the source's
    # group key. Used where the time exists but only somewhere a path cannot
    # reach it — burned into the image, say. The registration records where each
    # value came from; this only records what it is.
    start_timestamps: dict[str, str] = field(default_factory=dict)

    # -- matching -----------------------------------------------------------

    @property
    def fields(self) -> tuple[str, ...]:
        return tuple(_FIELD.findall(self.path))

    @property
    def glob(self) -> str:
        return _FIELD.sub("*", self.path)

    @property
    def regex(self) -> re.Pattern[str]:
        pattern = "".join(
            f"(?P<{part[1:-1]}>[^/]+)" if part.startswith("{") else re.escape(part)
            for part in re.split(r"(\{\w+\})", self.path)
            if part
        )
        return re.compile(f"^{pattern}$")

    def match(self, relative: str) -> dict[str, str] | None:
        found = self.regex.match(relative)
        return found.groupdict() if found else None

    # -- interpretation -----------------------------------------------------

    def camera_id(self, captures: dict[str, str]) -> str:
        raw = captures.get(self.camera_field, "")
        return self.camera_map.get(raw, raw)

    def day_key(self, captures: dict[str, str]) -> str:
        return captures.get(self.day_field, "")

    def identity(self, captures: dict[str, str]) -> str:
        return captures.get(self.identity_field, "") if self.identity_field else ""

    def timestamp(self, captures: dict[str, str]) -> datetime | None:
        """A capture time, or None. Never a substituted default."""
        if not self.timestamp_field or not self.timestamp_format:
            return None
        raw = captures.get(self.timestamp_field)
        if not raw:
            return None
        try:
            return datetime.strptime(raw, self.timestamp_format).replace(tzinfo=UTC)
        except ValueError:
            return None

    def declared_timestamp(self, captures: dict[str, str]) -> datetime | None:
        raw = self.start_timestamps.get("/".join(self.group_key(captures)))
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)

    def group_key(self, captures: dict[str, str]) -> tuple[str, ...]:
        keys = self.group_by or tuple(f for f in self.fields if f != self.order_by)
        return tuple(captures.get(k, "") for k in keys)

    def validate(self) -> None:
        if self.kind not in {"image_sequence", "video"}:
            raise RegistrationError(f"unknown layout kind {self.kind!r}")
        names = set(self.fields)
        if not names:
            raise RegistrationError(f"layout path {self.path!r} captures no fields")

        # camera and day carry defaults, because a single-camera source need not
        # capture either. A role the caller named explicitly must exist.
        for role, value in (
            ("group_by", self.group_by),
            ("order_by", (self.order_by,) if self.order_by else ()),
            ("identity_field", (self.identity_field,) if self.identity_field else ()),
            ("timestamp_field", (self.timestamp_field,) if self.timestamp_field else ()),
        ):
            for name in value:
                if name not in names:
                    raise RegistrationError(
                        f"layout names {role}={name!r}, which the path template "
                        f"{self.path!r} does not capture"
                    )


def layout_from_dict(data: dict[str, Any] | None, *, where: str) -> LayoutSpec | None:
    if not data:
        return None
    data = dict(data)
    try:
        spec = LayoutSpec(
            path=data.pop("path"),
            kind=data.pop("kind", "image_sequence"),
            group_by=tuple(data.pop("group_by", ()) or ()),
            order_by=data.pop("order_by", "") or "",
            camera_field=data.pop("camera_field", "camera") or "",
            day_field=data.pop("day_field", "day") or "",
            identity_field=data.pop("identity_field", "") or "",
            camera_map={str(k): str(v) for k, v in (data.pop("camera_map", {}) or {}).items()},
            timestamp_field=data.pop("timestamp_field", "") or "",
            timestamp_format=data.pop("timestamp_format", "") or "",
            start_timestamps={
                str(k): str(v) for k, v in (data.pop("start_timestamps", {}) or {}).items()
            },
        )
    except KeyError as exc:
        raise RegistrationError(f"{where}: layout is missing {exc}") from exc
    if data:
        raise RegistrationError(f"{where}: unknown layout keys {sorted(data)}")
    spec.validate()
    return spec


def relative_paths(root: Path, spec: LayoutSpec) -> list[str]:
    """Files beneath ``root`` that the layout's template matches."""
    return sorted(p.relative_to(root).as_posix() for p in root.glob(spec.glob) if p.is_file())


def sources_from_layout(registration, spec: LayoutSpec, root: Path):
    """Group the files a layout matches into registered sources.

    Split keys come from the registration; camera, day, identity and capture
    time come from the layout's captures. Nothing downstream needs to know which
    dataset produced them.
    """
    from ..ingest.source import register_source

    grouped: dict[tuple[str, ...], list[tuple[str, dict[str, str]]]] = {}
    for relative in relative_paths(root, spec):
        captures = spec.match(relative)
        if captures is None:
            continue
        grouped.setdefault(spec.group_key(captures), []).append((relative, captures))

    declared_cameras = set(registration.camera_ids)
    sources = []
    for key, members in sorted(grouped.items()):
        members.sort(key=lambda item: item[1].get(spec.order_by, item[0]))
        first_relative, first = members[0][0], members[0][1]

        # A layout that captures no camera falls back to the registration's
        # first declared camera, which is the honest reading of a single-camera
        # source.
        camera_id = spec.camera_id(first) or registration.camera_ids[0]
        if declared_cameras and camera_id not in declared_cameras:
            raise RegistrationError(
                f"{registration.name}: layout produced camera {camera_id!r}, which the "
                f"registration does not declare (declares: {sorted(declared_cameras)})",
                missing_field="camera_ids",
            )

        timestamp = spec.timestamp(first) or spec.declared_timestamp(first)
        day_key = spec.day_key(first)
        if not day_key and timestamp is not None:
            day_key = timestamp.date().isoformat()

        paths = [str(root / relative) for relative, _ in members]
        sources.append(
            register_source(
                source_id=f"{registration.name}/{'/'.join(key)}",
                camera_id=camera_id,
                site_key=registration.site_key,
                animal_set_key=registration.animal_set_keys[0],
                media_path=paths[0] if spec.kind == "video" else str(root / first_relative),
                media_paths=() if spec.kind == "video" else tuple(paths),
                dataset_name=registration.name,
                dataset_version=registration.version,
                day_key=day_key,
                start_timestamp=timestamp,
                kind="video" if spec.kind == "video" else "image_sequence",
                animal_id=spec.identity(first),
            )
        )
    return tuple(sources)
