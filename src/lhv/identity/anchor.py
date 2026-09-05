"""The external-anchor interface.

In production the anchor is a parlour, AMS or RFID identifier stream. No public
dataset carries one, so P0 routes the dataset's own ground-truth identity
through this same interface, marked with its anchor source. That exercises the
production control flow — anchor first, fallback second, unresolved as a real
outcome — without a farm, and makes the later substitution a change of source
rather than a change of plumbing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from .schemas import AnchorRecord

__all__ = ["AnchorSource", "InMemoryAnchorSource", "DatasetLabelAnchorSource"]


class AnchorSource(Protocol):
    """Anything that can say which animal was at a place during a window."""

    @property
    def source_name(self) -> str: ...

    def records_for(
        self, *, site_key: str, start: datetime, end: datetime, camera_id: str = ""
    ) -> list[AnchorRecord]: ...


class InMemoryAnchorSource:
    """An anchor source backed by records held in memory."""

    def __init__(self, records: list[AnchorRecord], *, source_name: str = "in-memory") -> None:
        self._records = list(records)
        self._source_name = source_name

    @property
    def source_name(self) -> str:
        return self._source_name

    def add(self, record: AnchorRecord) -> None:
        self._records.append(record)

    def records_for(
        self,
        *,
        site_key: str,
        start: datetime,
        end: datetime,
        camera_id: str = "",
        tolerance_seconds: float = 0.0,
    ) -> list[AnchorRecord]:
        return [
            record
            for record in self._records
            if record.site_key == site_key
            and record.overlaps(start, end, tolerance_seconds=tolerance_seconds)
            and (not camera_id or not record.camera_id or record.camera_id == camera_id)
        ]


class DatasetLabelAnchorSource(InMemoryAnchorSource):
    """A simulated anchor built from a dataset's ground-truth identity labels.

    The anchor source name records what it really is, so no report can present
    a dataset label as a farm identifier. Its accuracy is by construction
    perfect, which flatters the identity layer — the evaluation harness reports
    the visual fallback's standalone performance separately for exactly that
    reason.
    """

    def __init__(self, records: list[AnchorRecord], *, dataset_name: str) -> None:
        super().__init__(records, source_name=f"dataset-label:{dataset_name}")
        self.dataset_name = dataset_name

    @classmethod
    def from_labelled_tracklets(
        cls,
        labels: dict[str, str],
        tracklets,
        *,
        dataset_name: str,
    ) -> DatasetLabelAnchorSource:
        """Build anchors from a tracklet-id -> animal-id label mapping."""
        records: list[AnchorRecord] = []
        for tracklet in tracklets:
            animal_id = labels.get(tracklet.tracklet_id)
            if animal_id is None:
                continue
            if tracklet.first_timestamp is None or tracklet.last_timestamp is None:
                continue
            records.append(
                AnchorRecord(
                    animal_id=animal_id,
                    anchor_source=f"dataset-label:{dataset_name}",
                    site_key=tracklet.site_key,
                    observed_from=tracklet.first_timestamp,
                    observed_to=tracklet.last_timestamp,
                    camera_id=tracklet.camera_id,
                    reader_id="ground-truth",
                )
            )
        return cls(records, dataset_name=dataset_name)
