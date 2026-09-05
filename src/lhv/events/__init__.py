"""Stage boundary: risk scores -> exported health events.

The canonical versioned event schema, a named threshold policy, bounded evidence
retention with human masking, and a pluggable export adapter.
"""

from .build import EventBuilder
from .clips import ClipRetainer
from .export import (
    ConsumerUnavailable,
    DeliveryResult,
    EventExporter,
    ExportReport,
    FileExportAdapter,
)
from .masking import DetectorMasker, Masker, MaskingError, RegionMasker
from .policy import ThresholdPolicy
from .schemas import EventLevel, EvidenceClip, HealthEvent, ObservationWindow, PolicyIdentity

__all__ = [
    "ClipRetainer",
    "ConsumerUnavailable",
    "DeliveryResult",
    "DetectorMasker",
    "EventBuilder",
    "EventExporter",
    "EventLevel",
    "EvidenceClip",
    "ExportReport",
    "FileExportAdapter",
    "HealthEvent",
    "Masker",
    "MaskingError",
    "ObservationWindow",
    "PolicyIdentity",
    "RegionMasker",
    "ThresholdPolicy",
]
