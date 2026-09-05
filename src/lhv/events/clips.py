"""Bounded evidence retention.

An alert keeps a clip so a person can look at what raised it. A routine
observation keeps nothing, because retaining footage of every animal every day
is neither necessary for review nor defensible to keep.

Masking runs in memory, frame by frame, before the writer opens its file. If it
fails on any frame, the whole clip is abandoned and the event records that
retention failed. A partially masked clip is not a lesser good; it is the thing
being prevented.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ResolvedConfig
from ..ingest.decode import open_decoder
from ..ingest.isolation import assert_isolated_output_root
from .masking import Masker, MaskingError
from .schemas import EventLevel, EvidenceClip

__all__ = ["ClipRetainer"]


def _safe_name(value: str) -> str:
    import hashlib
    import re

    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")[:96]
    return f"{stem}-{hashlib.sha1(value.encode('utf-8')).hexdigest()[:10]}"


class ClipRetainer:
    """Writes a bounded, masked clip around an observation window."""

    def __init__(
        self,
        output_root: str | Path,
        config: ResolvedConfig,
        *,
        masker: Masker | None = None,
    ) -> None:
        self.output_root = assert_isolated_output_root(output_root)
        self.config = config
        self.masker = masker
        self.clips_written = 0
        self.retention_failures = 0

    def retain(
        self,
        *,
        level: EventLevel,
        event_id: str,
        media_path: str,
        media_kind: str,
        first_frame_index: int,
        last_frame_index: int,
        frame_rate: float,
    ) -> EvidenceClip | None:
        """Retain a clip for an alert. Anything else retains nothing."""
        if level is not EventLevel.ALERT:
            return None

        if self.masker is None:
            self.retention_failures += 1
            return EvidenceClip(
                retained=False,
                failure_reason=(
                    "no masker configured; a clip may not be written without human masking"
                ),
            )

        rate = frame_rate if frame_rate > 0 else 25.0
        before = int(self.config.events.clip_seconds_before * rate)
        after = int(self.config.events.clip_seconds_after * rate)
        start = max(first_frame_index - before, 0)
        end = last_frame_index + after

        try:
            frames = self._masked_frames(media_path, media_kind, start, end)
        except MaskingError as exc:
            self.retention_failures += 1
            return EvidenceClip(
                retained=False,
                seconds_before=self.config.events.clip_seconds_before,
                seconds_after=self.config.events.clip_seconds_after,
                masking_applied=False,
                masking_mode=self.masker.mode,
                failure_reason=f"masking failed, so no clip was retained: {exc}",
            )
        except (OSError, FileNotFoundError) as exc:
            self.retention_failures += 1
            return EvidenceClip(
                retained=False, failure_reason=f"source media could not be read: {exc}"
            )

        if not frames:
            self.retention_failures += 1
            return EvidenceClip(
                retained=False,
                masking_mode=self.masker.mode,
                failure_reason="no frames fell within the requested window",
            )

        try:
            path = self._write(event_id, frames, rate)
        except OSError as exc:
            self.retention_failures += 1
            return EvidenceClip(
                retained=False,
                masking_mode=self.masker.mode,
                masking_applied=True,
                failure_reason=f"the masked clip could not be written: {exc}",
            )
        self.clips_written += 1
        return EvidenceClip(
            retained=True,
            path=str(path),
            seconds_before=self.config.events.clip_seconds_before,
            seconds_after=self.config.events.clip_seconds_after,
            masking_applied=True,
            masking_mode=self.masker.mode,
            frame_count=len(frames),
        )

    def _masked_frames(self, media_path: str, media_kind: str, start: int, end: int) -> list:
        """Decode and mask entirely in memory. Nothing here touches storage."""
        decoder = open_decoder(media_path, kind=media_kind)
        masked = []
        try:
            for decoded in decoder.iter_frames(start_index=start, decode=True):
                if decoded.index > end:
                    break
                if decoded.image is None:
                    continue
                masked.append(self.masker.mask(decoded.image))
        finally:
            decoder.close()
        return masked

    def _write(self, event_id: str, frames: list, frame_rate: float) -> Path:
        import cv2

        directory = self.output_root / "clips"
        directory.mkdir(parents=True, exist_ok=True)
        # Event identifiers carry source paths and stage suffixes, neither of
        # which is a legal filename. The event still names the clip it owns
        # through the path recorded on it.
        path = directory / f"{_safe_name(event_id)}.avi"
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (width, height)
        )
        if not writer.isOpened():
            raise OSError(f"no usable video writer for {path}")
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return path
