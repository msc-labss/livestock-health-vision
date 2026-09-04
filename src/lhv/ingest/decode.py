"""Decoding backends.

Two shapes of source material exist in the public data this change targets:
video files and directories of ordered images. Both present the same interface,
so nothing above this module knows which it is reading.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

__all__ = ["DecodedFrame", "Decoder", "VideoFileDecoder", "ImageSequenceDecoder", "open_decoder"]

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DecodedFrame:
    """One decoded frame and the offset the container reported for it.

    ``offset_seconds`` is None when the container reported nothing usable. It is
    never filled in with a guess.
    """

    index: int
    image: np.ndarray | None
    offset_seconds: float | None


class Decoder(Protocol):
    @property
    def frame_count(self) -> int: ...

    @property
    def frame_rate(self) -> float: ...

    def iter_frames(
        self, *, start_index: int = 0, decode: bool = True
    ) -> Iterator[DecodedFrame]: ...

    def close(self) -> None: ...


class VideoFileDecoder:
    """Decodes a video file with OpenCV, reporting container timestamps as given."""

    def __init__(self, path: str | Path) -> None:
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"no such video: {self.path}")
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise OSError(f"could not open video: {self.path}")
        self._frame_count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_rate = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)

    @property
    def frame_count(self) -> int:
        return max(self._frame_count, 0)

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    def iter_frames(self, *, start_index: int = 0, decode: bool = True) -> Iterator[DecodedFrame]:
        cv2 = self._cv2
        if start_index:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
        index = start_index
        while True:
            ok, image = self._capture.read()
            if not ok:
                break
            # Read the position after the decode: that is the presentation time
            # of the frame just returned. Containers that carry no usable
            # timestamp report 0.0 here, which is only legitimate for the very
            # first frame — anywhere else it means "unknown", and unknown is
            # reported as unknown rather than filled in.
            position_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)
            if position_ms > 0:
                offset = position_ms / 1000.0
            elif index == 0 and position_ms == 0:
                offset = 0.0
            else:
                offset = None
            yield DecodedFrame(index=index, image=image if decode else None, offset_seconds=offset)
            index += 1

    def close(self) -> None:
        self._capture.release()


class ImageSequenceDecoder:
    """Decodes an ordered directory of images.

    Ordering is by filename so that iteration is reproducible; a container
    timestamp does not exist for this shape and is reported as absent rather
    than synthesised from the position in the sequence.
    """

    def __init__(self, path: str | Path, *, frame_rate: float = 0.0) -> None:
        import cv2

        self._cv2 = cv2
        self.path = Path(path)
        if not self.path.is_dir():
            raise NotADirectoryError(f"not an image-sequence directory: {self.path}")
        self._files = sorted(
            p for p in self.path.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        self._frame_rate = frame_rate

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(self._files)

    @property
    def frame_count(self) -> int:
        return len(self._files)

    @property
    def frame_rate(self) -> float:
        return self._frame_rate

    def iter_frames(self, *, start_index: int = 0, decode: bool = True) -> Iterator[DecodedFrame]:
        for index in range(start_index, len(self._files)):
            image = self._cv2.imread(str(self._files[index])) if decode else None
            offset = index / self._frame_rate if self._frame_rate > 0 else None
            yield DecodedFrame(index=index, image=image, offset_seconds=offset)

    def close(self) -> None:
        return None


def open_decoder(path: str | Path, *, kind: str, frame_rate: float = 0.0) -> Decoder:
    if kind == "image_sequence":
        return ImageSequenceDecoder(path, frame_rate=frame_rate)
    return VideoFileDecoder(path)
