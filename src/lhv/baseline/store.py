"""The per-animal time series, over partitioned columnar files.

Bulk records are append-only and never updated, which is what a columnar file is
for. Partitioning by site and day makes selective deletion cheap and makes a
day-disjoint or site-disjoint read a directory selection rather than a scan. The
embedded analytical database reads those same files in place, so there is one
copy of the data rather than two.

Three areas, deliberately separate:

* ``observations``  — valid passes with a resolved identity. The time series.
* ``unattributed``  — valid passes whose identity is unresolved. No animal's
  series is touched by these, but they are not thrown away.
* ``audit``         — invalid passes, retained with their reason and presented
  to nobody as measurements.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..phenotype.schemas import FeatureRecord
from ..profiles import FeatureSet

__all__ = ["Observation", "AppendResult", "TimeSeriesStore", "to_storage_time", "from_storage_time"]


def to_storage_time(value: datetime | None) -> datetime | None:
    """Normalise to naive UTC, which is how timestamps are stored.

    Storing a fixed offset keeps the columnar files portable and keeps the query
    engine from needing a timezone database to read them back. Every timestamp
    in the store is UTC; nothing else is admitted.

    None survives as None. A source that records no capture time still has its
    passes retained — in the unattributed area, never in a series — and writing
    them must not require inventing the time they were refused for lacking.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def from_storage_time(value: datetime | None) -> datetime | None:
    """Re-attach UTC on the way out, so callers never see a naive timestamp."""
    if value is None:
        return None
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


OBSERVATIONS = "observations"
UNATTRIBUTED = "unattributed"
AUDIT = "audit"


@dataclass(frozen=True)
class Observation:
    """One row of a per-animal time series."""

    animal_id: str
    pass_id: str
    observed_at: datetime | None
    site_key: str
    day_key: str
    feature_set_version: str
    features: dict[str, float]
    injected: bool = False
    tracklet_id: str = ""

    def value(self, name: str) -> float | None:
        return self.features.get(name)


@dataclass
class AppendResult:
    appended: int = 0
    unattributed: int = 0
    audited: int = 0
    skipped_without_timestamp: int = 0
    files_written: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"appended {self.appended} observation(s), {self.unattributed} unattributed, "
            f"{self.audited} retained for audit, "
            f"{self.skipped_without_timestamp} without a usable timestamp"
        )


class TimeSeriesStore:
    """Reads and writes the per-animal series and its two sidecars."""

    def __init__(self, root: str | Path, feature_set: FeatureSet) -> None:
        self.root = Path(root)
        self.feature_set = feature_set
        self.root.mkdir(parents=True, exist_ok=True)

    # -- layout -------------------------------------------------------------

    def area(self, name: str) -> Path:
        return self.root / name

    def _partition(self, area: str, site_key: str, day_key: str) -> Path:
        return self.area(area) / f"site_key={site_key}" / f"day_key={day_key}"

    def _schema(self) -> pa.Schema:
        fields = [
            pa.field("animal_id", pa.string()),
            pa.field("pass_id", pa.string()),
            pa.field("tracklet_id", pa.string()),
            pa.field("observed_at", pa.timestamp("us")),
            pa.field("feature_set_version", pa.string()),
            pa.field("injected", pa.bool_()),
            pa.field("valid", pa.bool_()),
            pa.field("validity_reason", pa.string()),
            pa.field("completeness", pa.string()),
        ]
        for name in self.feature_set.names:
            fields.append(pa.field(f"f_{name}", pa.float64()))
            fields.append(pa.field(f"q_{name}", pa.string()))
        return pa.schema(fields)

    # -- writing ------------------------------------------------------------

    def append(self, records: Iterable[FeatureRecord]) -> AppendResult:
        """Route each record to the area its validity and identity dictate."""
        result = AppendResult()
        batches: dict[tuple[str, str, str], list[FeatureRecord]] = {}

        for record in records:
            if not record.valid:
                area = AUDIT
                result.audited += 1
            elif not record.animal_id:
                area = UNATTRIBUTED
                result.unattributed += 1
            elif record.observed_at is None:
                # A series is keyed by observation time. Without one there is
                # nowhere in the series to put it, and inventing a time is how a
                # baseline silently becomes wrong.
                area = UNATTRIBUTED
                result.skipped_without_timestamp += 1
            else:
                area = OBSERVATIONS
                result.appended += 1
            batches.setdefault((area, record.site_key, record.day_key), []).append(record)

        for (area, site_key, day_key), group in sorted(batches.items()):
            directory = self._partition(area, site_key, day_key)
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"part-{uuid.uuid4().hex[:12]}.parquet"
            pq.write_table(self._to_table(group), path)
            result.files_written.append(str(path))

        return result

    def _to_table(self, records: list[FeatureRecord]) -> pa.Table:
        columns: dict[str, list] = {
            "animal_id": [],
            "pass_id": [],
            "tracklet_id": [],
            "observed_at": [],
            "feature_set_version": [],
            "injected": [],
            "valid": [],
            "validity_reason": [],
            "completeness": [],
        }
        for name in self.feature_set.names:
            columns[f"f_{name}"] = []
            columns[f"q_{name}"] = []

        for record in records:
            columns["animal_id"].append(record.animal_id)
            columns["pass_id"].append(record.pass_id)
            columns["tracklet_id"].append(record.tracklet_id)
            columns["observed_at"].append(to_storage_time(record.observed_at))
            columns["feature_set_version"].append(record.feature_set_version)
            columns["injected"].append(bool(record.injected))
            columns["valid"].append(bool(record.valid))
            columns["validity_reason"].append(str(record.validity_reason))
            columns["completeness"].append(str(record.completeness))

            emitted = {f.name: f for f in record.measurements()}
            for name in self.feature_set.names:
                feature = emitted.get(name)
                columns[f"f_{name}"].append(None if feature is None else float(feature.value))
                columns[f"q_{name}"].append(None if feature is None else str(feature.quality))

        return pa.Table.from_pydict(columns, schema=self._schema())

    # -- reading ------------------------------------------------------------

    def connect(self):
        """An embedded analytical database that reads the columnar files in place."""
        import duckdb

        return duckdb.connect()

    def _glob(self, area: str) -> str:
        return str(self.area(area) / "**" / "*.parquet")

    def has_data(self, area: str = OBSERVATIONS) -> bool:
        return any(self.area(area).rglob("*.parquet"))

    def read(
        self,
        *,
        area: str = OBSERVATIONS,
        animal_id: str | None = None,
        animal_ids: Iterable[str] | None = None,
        site_key: str | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        exclude_pass_ids: Iterable[str] | None = None,
    ) -> list[Observation]:
        """Query the series. Time bounds are inclusive of ``start``, exclusive of ``end``."""
        if not self.has_data(area):
            return []

        clauses: list[str] = []
        parameters: list = []
        if animal_id is not None:
            clauses.append("animal_id = ?")
            parameters.append(animal_id)
        if animal_ids is not None:
            ids = list(animal_ids)
            if not ids:
                return []
            clauses.append(f"animal_id IN ({', '.join('?' for _ in ids)})")
            parameters.extend(ids)
        if site_key is not None:
            clauses.append("site_key = ?")
            parameters.append(site_key)
        if start is not None:
            clauses.append("observed_at >= ?")
            parameters.append(to_storage_time(start))
        if end is not None:
            clauses.append("observed_at < ?")
            parameters.append(to_storage_time(end))
        excluded = list(exclude_pass_ids or [])
        if excluded:
            clauses.append(f"pass_id NOT IN ({', '.join('?' for _ in excluded)})")
            parameters.extend(excluded)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT * FROM read_parquet(?, hive_partitioning = true) {where} "
            f"ORDER BY observed_at, pass_id"
        )
        with self.connect() as connection:
            cursor = connection.execute(query, [self._glob(area), *parameters])
            names = [description[0] for description in cursor.description]
            rows = cursor.fetchall()

        return [self._to_observation(dict(zip(names, row, strict=True))) for row in rows]

    def _to_observation(self, row: dict) -> Observation:
        features = {}
        for name in self.feature_set.names:
            value = row.get(f"f_{name}")
            if value is not None:
                features[name] = float(value)
        return Observation(
            animal_id=row["animal_id"],
            pass_id=row["pass_id"],
            observed_at=from_storage_time(row["observed_at"]),
            site_key=row["site_key"],
            day_key=row["day_key"],
            feature_set_version=row["feature_set_version"],
            features=features,
            injected=bool(row.get("injected")),
            tracklet_id=row.get("tracklet_id") or "",
        )

    def animal_ids(self, *, site_key: str | None = None) -> tuple[str, ...]:
        if not self.has_data():
            return ()
        clause = "WHERE site_key = ?" if site_key else ""
        parameters = [self._glob(OBSERVATIONS)] + ([site_key] if site_key else [])
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT DISTINCT animal_id FROM read_parquet(?, hive_partitioning = true) "
                f"{clause} ORDER BY animal_id",
                parameters,
            ).fetchall()
        return tuple(row[0] for row in rows)
