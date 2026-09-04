"""Export.

There is no farm database to integrate with yet, so P0 ships a file-backed
adapter behind the interface P1 will substitute a transport into. What is fixed
now is the delivery contract, which is the part that is expensive to change
later: at-least-once delivery with an idempotency key, so a retry cannot become
a duplicate health event downstream, and undelivered events are retained and
counted rather than dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .schemas import HealthEvent

__all__ = [
    "DeliveryResult",
    "ExportAdapter",
    "FileExportAdapter",
    "ExportReport",
    "EventExporter",
]


class ConsumerUnavailable(RuntimeError):
    """The consumer could not be reached. The event is retained and retried."""


@dataclass(frozen=True)
class DeliveryResult:
    key: str
    delivered: bool
    duplicate: bool = False
    location: str = ""


class ExportAdapter(Protocol):
    @property
    def name(self) -> str: ...

    def deliver(self, event: HealthEvent) -> DeliveryResult: ...

    def seen(self, key: str) -> bool: ...


class FileExportAdapter:
    """Writes each event as a JSON document named by its idempotency key.

    A manifest of delivered keys is what makes redelivery detectable: the second
    delivery of a key is reported as a duplicate and does not produce a second
    event.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "manifest.jsonl"
        self.available = True

    @property
    def name(self) -> str:
        return f"file:{self.directory}"

    def seen(self, key: str) -> bool:
        return (self.directory / f"{key}.json").exists()

    def delivered_keys(self) -> set[str]:
        if not self.manifest_path.exists():
            return set()
        keys = set()
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                keys.add(json.loads(line)["idempotency_key"])
        return keys

    def deliver(self, event: HealthEvent) -> DeliveryResult:
        if not self.available:
            raise ConsumerUnavailable(f"{self.name} is unavailable")

        key = event.idempotency_key
        target = self.directory / f"{key}.json"
        if target.exists():
            return DeliveryResult(key=key, delivered=True, duplicate=True, location=str(target))

        target.write_text(json.dumps(event.to_dict(), indent=2, default=str), encoding="utf-8")
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "idempotency_key": key,
                        "event_id": event.event_id,
                        "schema_name": event.schema_name,
                        "schema_version": event.schema_version,
                        "level": str(event.level),
                    }
                )
                + "\n"
            )
        return DeliveryResult(key=key, delivered=True, location=str(target))


@dataclass
class ExportReport:
    attempted: int = 0
    delivered: int = 0
    duplicates: int = 0
    undelivered: int = 0
    retries: int = 0
    pending_keys: list[str] = field(default_factory=list)

    def describe(self) -> str:
        return (
            f"export: {self.delivered} delivered ({self.duplicates} already present), "
            f"{self.undelivered} undelivered after {self.retries} retry attempt(s)"
        )


class EventExporter:
    """Delivers events, retaining and retrying anything the consumer refused."""

    def __init__(
        self,
        adapter: ExportAdapter,
        *,
        retention_dir: str | Path,
        max_attempts: int = 3,
    ) -> None:
        self.adapter = adapter
        self.retention_dir = Path(retention_dir)
        self.retention_dir.mkdir(parents=True, exist_ok=True)
        self.max_attempts = max_attempts
        self.report = ExportReport()

    def _retain(self, event: HealthEvent) -> Path:
        path = self.retention_dir / f"{event.idempotency_key}.json"
        path.write_text(json.dumps(event.to_dict(), indent=2, default=str), encoding="utf-8")
        return path

    def _release(self, key: str) -> None:
        path = self.retention_dir / f"{key}.json"
        if path.exists():
            path.unlink()

    def pending(self) -> list[str]:
        return sorted(p.stem for p in self.retention_dir.glob("*.json"))

    def export(self, events) -> ExportReport:
        for event in events:
            self.report.attempted += 1
            self._deliver_with_retry(event)
        self.report.pending_keys = self.pending()
        return self.report

    def _deliver_with_retry(self, event: HealthEvent) -> None:
        retained = self._retain(event)
        for attempt in range(self.max_attempts):
            if attempt:
                self.report.retries += 1
            try:
                result = self.adapter.deliver(event)
            except ConsumerUnavailable:
                continue
            if result.delivered:
                self.report.delivered += 1
                if result.duplicate:
                    self.report.duplicates += 1
                self._release(result.key)
                return
        # Every attempt failed. The event stays on disk so a later run can
        # deliver it; it is never dropped to make a run look clean.
        self.report.undelivered += 1
        assert retained.exists()

    def retry_pending(self, events_by_key: dict[str, HealthEvent]) -> ExportReport:
        """Deliver whatever a previous run retained."""
        for key in self.pending():
            event = events_by_key.get(key)
            if event is None:
                continue
            self.report.attempted += 1
            self._deliver_with_retry(event)
        self.report.pending_keys = self.pending()
        return self.report
