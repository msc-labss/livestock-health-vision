"""Stage boundary: recorded video -> frames carrying provenance.

Turns a registered source into a deterministic, resumable frame stream in which
every frame carries the provenance and split keys that every later stage and the
evaluation harness depend on.
"""

from .decode import DecodedFrame, ImageSequenceDecoder, VideoFileDecoder, open_decoder
from .isolation import assert_isolated_output_root, is_tracked_by_version_control
from .provenance import FrameProvenance, SourceProvenance
from .source import RegisteredSource, register_source, register_sources_from_dataset
from .stream import Frame, IngestCheckpoint, Ingestor, IngestReport, TimestampAnomaly

__all__ = [
    "DecodedFrame",
    "Frame",
    "FrameProvenance",
    "ImageSequenceDecoder",
    "IngestCheckpoint",
    "IngestReport",
    "Ingestor",
    "RegisteredSource",
    "SourceProvenance",
    "TimestampAnomaly",
    "VideoFileDecoder",
    "assert_isolated_output_root",
    "is_tracked_by_version_control",
    "open_decoder",
    "register_source",
    "register_sources_from_dataset",
]
