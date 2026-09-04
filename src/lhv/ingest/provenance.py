"""Frame provenance — the record every later stage reads instead of guessing.

Provenance travels with the frame rather than being reconstructed downstream,
because reconstruction is where split keys get lost and evaluation quietly
leaks. The split keys live here for the same reason: the harness must be able to
partition on animal set, day and site without touching the media again.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from ..schema import Record, opt, req

__all__ = ["FrameProvenance", "SourceProvenance"]


@dataclass(frozen=True)
class SourceProvenance(Record):
    """The registered identity of one piece of source material."""

    SCHEMA_NAME = "source_provenance"
    SCHEMA_VERSION = "1"

    source_id: str = req()
    camera_id: str = req()
    site_key: str = req()
    animal_set_key: str = req()
    dataset_name: str = req()
    dataset_version: str = req()
    media_path: str = opt("")
    frame_rate: float = opt(0.0)
    frame_count: int = opt(0)


@dataclass(frozen=True)
class FrameProvenance(Record):
    """Where a frame came from, and the keys evaluation will split on.

    ``capture_timestamp`` is never defaulted. When it cannot be recovered the
    frame carries ``timestamp_reliable = False`` and downstream stages filter on
    that mark.
    """

    SCHEMA_NAME = "frame_provenance"
    SCHEMA_VERSION = "1"

    source_id: str = req()
    camera_id: str = req()
    frame_index: int = req()
    site_key: str = req()
    animal_set_key: str = req()
    day_key: str = req()
    capture_timestamp: datetime | None = opt(None)
    timestamp_reliable: bool = opt(True)
    dataset_name: str = opt("")
    dataset_version: str = opt("")
    config_digest: str = opt("")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative, got {self.frame_index}")

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_id, self.frame_index)

    def split_keys(self) -> dict[str, str]:
        """The keys the evaluation harness constructs disjoint partitions from."""
        return {
            "animal_set": self.animal_set_key,
            "day": self.day_key,
            "site": self.site_key,
            "camera": self.camera_id,
            "source": self.source_id,
        }
