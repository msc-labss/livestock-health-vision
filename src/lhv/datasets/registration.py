"""Dataset registration: the keys a source must supply before it can be ingested.

A registration is what turns a folder of media into something the pipeline will
accept. It carries the split keys every later stage and the evaluation harness
depend on — site, cameras, animal sets — together with the licence and the
access route under which the material was obtained.

Content counts are recorded twice on purpose: what the literature reports, and
what was actually found on disk. Only the second is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..errors import RegistrationError
from ..schema import Record, opt, req
from .layout import LayoutSpec, layout_from_dict, sources_from_layout

__all__ = [
    "AccessTerms",
    "LayoutSpec",
    "CountSpec",
    "CountVerification",
    "VerificationReport",
    "DatasetRegistration",
    "load_registration",
    "available_registrations",
    "REGISTRATION_DIR",
]

REGISTRATION_DIR = Path(__file__).parent / "registrations"


@dataclass(frozen=True)
class AccessTerms:
    """The licence and the route by which the material was obtained."""

    licence: str
    licence_url: str = ""
    # "open-download" needs nothing; "request-form" and "signed-agreement" both
    # need a human, and the pipeline must say so rather than fail obscurely.
    access_route: str = "open-download"
    request_url: str = ""
    download_url: str = ""
    commercial_use: str = "unknown"
    attribution: str = ""
    citation: str = ""
    recorded_on: str = ""
    notes: str = ""

    @property
    def requires_human_request(self) -> bool:
        return self.access_route != "open-download"


@dataclass(frozen=True)
class CountSpec:
    """How to count one kind of content on disk, so a claim can be checked.

    ``kind`` is "files", "dirs", or "video_frames". The last decodes every
    matching video and sums the frames it actually contains, which is the only
    way to check a frame count claimed for a release that ships video rather
    than stills.
    """

    kind: str
    glob: str
    declared: int | None = None
    note: str = ""


@dataclass(frozen=True)
class CountVerification:
    name: str
    declared: int | None
    observed: int
    matches: bool


@dataclass(frozen=True)
class VerificationReport:
    dataset: str
    root: str
    counts: tuple[CountVerification, ...]
    present: bool

    @property
    def ok(self) -> bool:
        return self.present and all(c.matches for c in self.counts)

    def describe(self) -> str:
        if not self.present:
            return f"{self.dataset}: not present at {self.root}"
        lines = [f"{self.dataset}: verified against {self.root}"]
        for count in self.counts:
            declared = "not declared" if count.declared is None else str(count.declared)
            mark = "ok" if count.matches else "MISMATCH"
            lines.append(f"  {count.name}: observed {count.observed}, declared {declared} [{mark}]")
        return "\n".join(lines)


@dataclass(frozen=True)
class DatasetRegistration(Record):
    """A registered dataset. Refused if any key a later stage needs is absent."""

    SCHEMA_NAME = "dataset_registration"
    SCHEMA_VERSION = "1"

    name: str = req()
    version: str = req()
    site_key: str = req()
    camera_ids: tuple[str, ...] = req()
    animal_set_keys: tuple[str, ...] = req()
    access: AccessTerms = req()
    root: str = opt("")
    description: str = opt("")
    count_specs: dict[str, CountSpec] = opt({})
    media_extensions: tuple[str, ...] = opt((".mp4", ".avi", ".mkv", ".mov", ".jpg", ".png"))
    limitations: tuple[str, ...] = opt(())
    # Named layouts, each describing one component of the release.
    layouts: dict[str, LayoutSpec] = opt({})

    def __post_init__(self) -> None:
        super().__post_init__()
        # A key that is present but empty is the same absence with extra steps.
        for name in ("site_key", "version", "name"):
            if not str(getattr(self, name)).strip():
                raise RegistrationError(
                    f"dataset registration: {name!r} is empty", missing_field=name
                )
        for name in ("camera_ids", "animal_set_keys"):
            if len(getattr(self, name)) == 0:
                raise RegistrationError(
                    f"dataset registration {self.name!r}: {name!r} is empty; downstream splits "
                    f"cannot be constructed without it",
                    missing_field=name,
                )

    # -- verification -------------------------------------------------------

    def local_root(self, data_root: str | Path | None = None) -> Path:
        if data_root is not None:
            return Path(data_root)
        return Path(self.root)

    def layout(self, name: str) -> LayoutSpec:
        try:
            return self.layouts[name]
        except KeyError as exc:
            available = ", ".join(sorted(self.layouts)) or "none"
            raise RegistrationError(
                f"{self.name}: no layout named {name!r} (declares: {available})"
            ) from exc

    def sources(self, name: str, data_root: str | Path | None = None):
        """Registered sources for one declared layout."""
        root = self.local_root(data_root)
        if not root.exists():
            raise RegistrationError(f"{self.name} is registered but not present at {root}")
        return sources_from_layout(self, self.layout(name), root)

    def verify(self, data_root: str | Path | None = None) -> VerificationReport:
        """Count what is actually on disk and compare against the declared figures.

        The declared figures come from the literature. This is the only thing
        that turns them into evidence.
        """
        root = self.local_root(data_root)
        if not root.exists():
            return VerificationReport(self.name, str(root), (), present=False)

        verifications = []
        for name, spec in sorted(self.count_specs.items()):
            if spec.kind == "dirs":
                observed = sum(1 for p in root.glob(spec.glob) if p.is_dir())
            elif spec.kind == "video_frames":
                observed = _count_video_frames(sorted(root.glob(spec.glob)))
            else:
                observed = sum(1 for p in root.glob(spec.glob) if p.is_file())
            verifications.append(
                CountVerification(
                    name=name,
                    declared=spec.declared,
                    observed=observed,
                    matches=spec.declared is None or spec.declared == observed,
                )
            )
        return VerificationReport(self.name, str(root), tuple(verifications), present=True)


# -- loading ----------------------------------------------------------------


def _count_video_frames(paths) -> int:
    """Frames actually decodable from each video, summed.

    The container's own frame count is trusted where it is plausible; where it
    is absent or nonsensical the frames are counted by decoding, because a
    header figure is a claim like any other.
    """
    import cv2

    total = 0
    for path in paths:
        if not path.is_file():
            continue
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                continue
            declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if declared > 0:
                total += declared
                continue
            counted = 0
            while capture.grab():
                counted += 1
            total += counted
        finally:
            capture.release()
    return total


def registration_from_dict(
    data: dict[str, Any], *, where: str = "<registration>"
) -> DatasetRegistration:
    data = dict(data)

    access_spec = data.pop("access", None)
    if access_spec is None:
        raise RegistrationError(
            f"{where}: missing required key 'access' (licence and access terms)",
            missing_field="access",
        )
    try:
        access = AccessTerms(**access_spec)
    except TypeError as exc:
        raise RegistrationError(f"{where}: malformed access terms: {exc}") from exc

    layouts = {
        name: layout_from_dict(spec, where=f"{where}.layouts.{name}")
        for name, spec in (data.pop("layouts", {}) or {}).items()
    }

    count_specs = {}
    for name, spec in (data.pop("counts", {}) or {}).items():
        try:
            count_specs[name] = CountSpec(**spec)
        except TypeError as exc:
            raise RegistrationError(f"{where}: malformed count spec {name!r}: {exc}") from exc

    for key in ("camera_ids", "animal_set_keys", "media_extensions", "limitations"):
        if key in data and data[key] is not None:
            data[key] = tuple(data[key])

    try:
        return DatasetRegistration(access=access, count_specs=count_specs, layouts=layouts, **data)
    except TypeError as exc:
        raise RegistrationError(f"{where}: malformed registration: {exc}") from exc


def load_registration(name: str, *, directory: Path | None = None) -> DatasetRegistration:
    directory = directory or REGISTRATION_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(available_registrations(directory=directory)) or "none"
        raise RegistrationError(
            f"no dataset registration named {name!r} (available: {available})",
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return registration_from_dict(data, where=str(path))


def available_registrations(*, directory: Path | None = None) -> tuple[str, ...]:
    directory = directory or REGISTRATION_DIR
    if not directory.exists():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.yaml")))
