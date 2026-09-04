"""Durable records at every stage boundary.

Each stage reads durable records and writes durable records, so recomputing a
baseline never requires recomputing detections. This is where those records
live: partitioned columnar files, one directory per stage, partitioned by site
and day so that selective deletion and a day- or site-scoped read are directory
operations rather than scans.

Each row carries typed partition and lookup columns alongside the record itself,
serialised with the schema name and version it declares. Keeping the record
whole is deliberate: a stage boundary is a versioned contract, and flattening it
into columns would let a schema change pass unnoticed. The analytical queries
this change actually runs — baselines, herd windows, evaluation — are over the
per-animal time series, which is typed and columnar in the ordinary way.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .ingest.isolation import assert_isolated_output_root
from .schema import Record

__all__ = ["StageStore"]

_SCHEMA = pa.schema(
    [
        pa.field("record_id", pa.string()),
        pa.field("schema_name", pa.string()),
        pa.field("schema_version", pa.string()),
        pa.field("source_id", pa.string()),
        pa.field("sort_key", pa.int64()),
        pa.field("payload", pa.string()),
    ]
)


class StageStore:
    """Reads and writes one run's stage records."""

    def __init__(self, root: str | Path, *, isolate: bool = True) -> None:
        self.root = assert_isolated_output_root(root) if isolate else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str) -> Path:
        return self.root / "stages" / stage

    def clear(self, stage: str) -> None:
        """Remove a stage's records, so re-running it replaces rather than appends."""
        import shutil

        directory = self.stage_dir(stage)
        if directory.exists():
            shutil.rmtree(directory)

    def write(
        self,
        stage: str,
        records: Iterable[Record],
        *,
        site_of=None,
        day_of=None,
        source_of=None,
        sort_of=None,
    ) -> list[Path]:
        records = list(records)
        if not records:
            return []

        site_of = site_of or (lambda r: getattr(r, "site_key", "") or "unknown")
        day_of = day_of or (lambda r: getattr(r, "day_key", "") or "unknown")
        source_of = source_of or (lambda r: getattr(r, "source_id", ""))
        sort_of = sort_of or (lambda r: 0)

        groups: dict[tuple[str, str], list[Record]] = {}
        for record in records:
            groups.setdefault((site_of(record), day_of(record)), []).append(record)

        written: list[Path] = []
        for (site_key, day_key), group in sorted(groups.items()):
            directory = self.stage_dir(stage) / f"site_key={site_key}" / f"day_key={day_key}"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{uuid.uuid4().hex[:12]}.parquet"
            table = pa.Table.from_pydict(
                {
                    "record_id": [_identifier(r) for r in group],
                    "schema_name": [r.SCHEMA_NAME for r in group],
                    "schema_version": [r.SCHEMA_VERSION for r in group],
                    "source_id": [str(source_of(r)) for r in group],
                    "sort_key": [int(sort_of(r)) for r in group],
                    "payload": [json.dumps(r.to_dict(), default=str) for r in group],
                },
                schema=_SCHEMA,
            )
            pq.write_table(table, path)
            written.append(path)
        return written

    def read(
        self,
        stage: str,
        klass: type[Record],
        *,
        site_key: str | None = None,
        day_key: str | None = None,
        source_id: str | None = None,
    ) -> list[Record]:
        directory = self.stage_dir(stage)
        if not directory.exists():
            return []

        pattern = directory
        if site_key is not None:
            pattern = pattern / f"site_key={site_key}"
            if day_key is not None:
                pattern = pattern / f"day_key={day_key}"

        rows: list[tuple[int, str, dict]] = []
        for path in sorted(pattern.rglob("*.parquet")):
            table = pq.read_table(path)
            for record_id, source, sort_key, payload in zip(
                table.column("record_id").to_pylist(),
                table.column("source_id").to_pylist(),
                table.column("sort_key").to_pylist(),
                table.column("payload").to_pylist(),
                strict=True,
            ):
                if source_id is not None and source != source_id:
                    continue
                rows.append((sort_key, record_id, json.loads(payload)))

        rows.sort(key=lambda row: (row[0], row[1]))
        return [klass.from_dict(payload) for _, _, payload in rows]

    def stages(self) -> Sequence[str]:
        directory = self.root / "stages"
        if not directory.exists():
            return ()
        return tuple(sorted(p.name for p in directory.iterdir() if p.is_dir()))


def _identifier(record: Record) -> str:
    for attribute in (
        "detection_id",
        "tracklet_id",
        "pose_id",
        "pass_id",
        "assessment_id",
        "event_id",
        "source_id",
    ):
        value = getattr(record, attribute, None)
        if value:
            return str(value)
    return uuid.uuid4().hex
