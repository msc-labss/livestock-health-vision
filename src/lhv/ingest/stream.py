"""Frame emission: deterministic, resumable, provenance-carrying.

Iteration order and content are a function of the source and the ingest
configuration and of nothing else, so two runs agree and a resumed run continues
rather than repeats.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..config import ResolvedConfig
from ..schema import Record, opt, req
from .decode import open_decoder
from .isolation import assert_isolated_output_root
from .provenance import FrameProvenance
from .source import RegisteredSource

__all__ = ["Frame", "IngestCheckpoint", "TimestampAnomaly", "IngestReport", "Ingestor"]


@dataclass
class Frame:
    """A decoded frame and its provenance.

    The provenance is the durable record; the pixels are not, and are dropped as
    soon as the stage that needs them is done.
    """

    provenance: FrameProvenance
    image: np.ndarray | None = None

    @property
    def index(self) -> int:
        return self.provenance.frame_index

    @property
    def frame_id(self) -> str:
        return f"{self.provenance.source_id}#{self.provenance.frame_index}"


@dataclass(frozen=True)
class IngestCheckpoint(Record):
    """The recorded position a run resumes from."""

    SCHEMA_NAME = "ingest_checkpoint"
    SCHEMA_VERSION = "1"

    source_id: str = req()
    last_emitted_index: int = req()
    config_digest: str = req()
    emitted: int = opt(0)

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def read(cls, path: str | Path) -> IngestCheckpoint:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class TimestampAnomaly:
    frame_index: int
    previous: str
    current: str
    kind: str = "non_monotonic"


@dataclass
class IngestReport:
    """What a run emitted, and what it refused to pretend about timestamps."""

    source_id: str = ""
    frames_emitted: int = 0
    frames_excluded: int = 0
    unreliable_timestamp_indices: list[int] = field(default_factory=list)
    timestamp_anomalies: list[TimestampAnomaly] = field(default_factory=list)
    first_index: int | None = None
    last_index: int | None = None

    def describe(self) -> str:
        lines = [
            f"{self.source_id}: emitted {self.frames_emitted} frame(s)"
            f"{f' [{self.first_index}..{self.last_index}]' if self.frames_emitted else ''}"
        ]
        if self.frames_excluded:
            lines.append(f"  excluded {self.frames_excluded} frame(s) by timestamp policy")
        if self.unreliable_timestamp_indices:
            lines.append(
                f"  {len(self.unreliable_timestamp_indices)} frame(s) carry an unreliable "
                f"timestamp; first at index {self.unreliable_timestamp_indices[0]}"
            )
        for anomaly in self.timestamp_anomalies:
            lines.append(
                f"  non-monotonic timestamp at frame {anomaly.frame_index}: "
                f"{anomaly.previous} -> {anomaly.current}"
            )
        return "\n".join(lines)


class Ingestor:
    """Turns one registered source into a provenance-carrying frame stream."""

    def __init__(
        self,
        source: RegisteredSource,
        config: ResolvedConfig,
        *,
        output_root: str | Path | None = None,
    ) -> None:
        self.source = source
        self.config = config
        # Checked before anything is decoded, let alone written.
        self.output_root = (
            assert_isolated_output_root(output_root) if output_root is not None else None
        )
        self.report = IngestReport(source_id=source.source_id)

    # -- positions ----------------------------------------------------------

    def checkpoint_path(self) -> Path:
        if self.output_root is None:
            raise ValueError("no output root configured, so no checkpoint can be recorded")
        safe = self.source.source_id.replace("/", "__")
        return self.output_root / "checkpoints" / f"{safe}.json"

    def record_position(self, index: int, emitted: int) -> Path:
        return IngestCheckpoint(
            source_id=self.source.source_id,
            last_emitted_index=index,
            config_digest=self.config.digest,
            emitted=emitted,
        ).write(self.checkpoint_path())

    def recorded_position(self) -> IngestCheckpoint | None:
        path = self.checkpoint_path()
        return IngestCheckpoint.read(path) if path.exists() else None

    # -- emission -----------------------------------------------------------

    def iter_frames(
        self,
        *,
        resume_from: int | None = None,
        decode: bool = True,
    ) -> Iterator[Frame]:
        """Emit frames in source order with full provenance attached.

        ``resume_from`` is the index of the last frame already emitted; emission
        starts at the frame after it.
        """
        ingest = self.config.ingest
        stride = max(int(ingest.frame_stride), 1)
        self.report = IngestReport(source_id=self.source.source_id)

        start_index = 0 if resume_from is None else resume_from + 1
        # Keep the stride phase anchored to the start of the source, so resuming
        # cannot shift which frames a run would have emitted.
        if start_index % stride:
            start_index += stride - (start_index % stride)

        decoder = open_decoder(
            self.source.media_path,
            kind=self.source.kind,
            frame_rate=self._declared_frame_rate(),
            files=self.source.media_paths or None,
        )
        previous_timestamp = None
        emitted = 0
        try:
            for decoded in decoder.iter_frames(start_index=start_index, decode=decode):
                if decoded.index % stride:
                    continue
                if ingest.max_frames is not None and emitted >= ingest.max_frames:
                    break

                timestamp, reliable = self._timestamp_for(decoded.offset_seconds)
                if not reliable:
                    if ingest.unreliable_timestamp_policy == "exclude":
                        self.report.frames_excluded += 1
                        continue
                    self.report.unreliable_timestamp_indices.append(decoded.index)
                elif previous_timestamp is not None and timestamp < previous_timestamp:
                    self.report.timestamp_anomalies.append(
                        TimestampAnomaly(
                            frame_index=decoded.index,
                            previous=previous_timestamp.isoformat(),
                            current=timestamp.isoformat(),
                        )
                    )
                if reliable:
                    previous_timestamp = timestamp

                provenance = FrameProvenance(
                    source_id=self.source.source_id,
                    camera_id=self.source.camera_id,
                    frame_index=decoded.index,
                    site_key=self.source.site_key,
                    animal_set_key=self.source.animal_set_key,
                    day_key=(
                        timestamp.date().isoformat() if reliable else self.source.default_day_key
                    ),
                    capture_timestamp=timestamp,
                    timestamp_reliable=reliable,
                    dataset_name=self.source.dataset_name,
                    dataset_version=self.source.dataset_version,
                    config_digest=self.config.digest,
                )
                if self.report.first_index is None:
                    self.report.first_index = decoded.index
                self.report.last_index = decoded.index
                self.report.frames_emitted += 1
                emitted += 1
                yield Frame(provenance=provenance, image=decoded.image)
        finally:
            decoder.close()

    def _declared_frame_rate(self) -> float:
        return 0.0

    def _timestamp_for(self, offset_seconds: float | None):
        """Resolve a capture timestamp, or say plainly that it is unavailable."""
        start = self.source.start_timestamp
        if start is None or offset_seconds is None:
            return None, False
        if offset_seconds < 0:
            return None, False
        return start + timedelta(seconds=offset_seconds), True
