"""The resolved configuration and its digest.

Every artefact records a digest over the resolved configuration, the model
identities and versions, and the dataset version. That digest is the single
value answering "is this the same run?", so it must change when anything
material changes and must not change when something immaterial is merely
reordered or renamed.

Material fields participate in the digest. Operational fields — where output
lands, which device runs the model, how many workers — do not, because they
change the cost of a run without changing its result.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError
from .schema import canonical_json

__all__ = [
    "ModelIdentity",
    "IngestConfig",
    "PerceptionConfig",
    "IdentityConfig",
    "PhenotypeConfig",
    "BaselineConfig",
    "EventsConfig",
    "EvaluationConfig",
    "RunConfig",
    "ResolvedConfig",
]

_IMMATERIAL = {"material": False}


@dataclass(frozen=True)
class ModelIdentity:
    """Identity and version of a pretrained model, recorded on every output it produces."""

    name: str
    version: str
    task: str
    weights_uri: str = ""
    weights_sha256: str = ""
    licence: str = "unknown"

    def __str__(self) -> str:
        return f"{self.name}@{self.version}"


@dataclass(frozen=True)
class IngestConfig:
    frame_stride: int = 1
    max_frames: int | None = None
    # "flag" keeps a frame whose timestamp is unrecoverable and marks it unreliable;
    # "exclude" drops it. Never substitute a default timestamp.
    unreliable_timestamp_policy: str = "flag"
    decode_backend: str = "opencv"


@dataclass(frozen=True)
class PerceptionConfig:
    detection_threshold: float = 0.25
    detection_low_confidence_threshold: float = 0.50
    pose_low_confidence_threshold: float = 0.50
    keypoint_visibility_threshold: float = 0.30
    # Which implementation runs behind each stage interface. Material, because
    # results produced by different backends are different results.
    detector_backend: str = "ultralytics"
    pose_backend: str = "ultralytics"
    max_track_gap_frames: int = 15
    min_track_iou: float = 0.20
    occlusion_gap_frames: int = 3
    device: str = field(default="auto", metadata=_IMMATERIAL)
    batch_size: int = field(default=8, metadata=_IMMATERIAL)


@dataclass(frozen=True)
class IdentityConfig:
    confidence_floor: float = 0.60
    anchor_window_tolerance_seconds: float = 2.0
    reid_similarity_floor: float = 0.55
    # A floor on the margin over the runner-up. Left at zero it gates nothing,
    # which is the honest default: the right value depends on the embedding, and
    # on an uncalibrated one no value helps. Measured against MultiCamCows2024,
    # raising this from 0 to 0.0037 moves the visual fallback's precision from
    # 0.18 to 0.55 while keeping 16% of its correct answers.
    reid_margin_floor: float = 0.0
    # How close to the best-agreeing located reading a rival must be to stay a
    # candidate. At 1.0 only the single best survives; lower keeps genuine ties
    # ambiguous rather than picking one.
    region_agreement_ratio: float = 0.90
    reid_embedding_dim: int = 128


@dataclass(frozen=True)
class PhenotypeConfig:
    # Boundaries are fractions of the frame along the configured axis, because
    # public sources do not share a lane geometry.
    boundary_axis: str = "y"
    entry_boundary: float = 0.20
    exit_boundary: float = 0.80
    min_pass_frames: int = 8
    # A pass is invalid when more than this fraction of its features are of
    # reduced quality.
    max_reduced_quality_fraction: float = 0.40
    # How many features must be computable at all for a pass to be worth
    # scoring. Counted over features the source could supply, not over every
    # feature the profile declares: a top-down view cannot see a cow's paws in
    # any pass, and holding that against each pass individually would reject
    # every one of them for a property of the camera.
    min_usable_features: int = 4
    # A partial pass has no observed entry or exit crossing, so its stride and
    # speed features are not comparable with a complete one. Admitting them is a
    # declared decision rather than a silent default.
    accept_partial_passes: bool = False


@dataclass(frozen=True)
class BaselineConfig:
    lookback_days: int = 21
    min_observations: int = 5
    herd_window_days: int = 3
    herd_min_animals: int = 3
    risk_scale: str = "robust_z"


@dataclass(frozen=True)
class EventsConfig:
    threshold_policy_name: str = "p0-default"
    threshold_policy_version: str = "1"
    alert_threshold: float = 3.0
    watch_threshold: float = 2.0
    clip_seconds_before: float = 3.0
    clip_seconds_after: float = 3.0
    masking_mode: str = "blur"
    export_adapter: str = "file"


@dataclass(frozen=True)
class EvaluationConfig:
    split_kind: str = "animal"
    test_fraction: float = 0.30
    detection_iou_threshold: float = 0.50
    keypoint_distance_threshold: float = 0.10


@dataclass(frozen=True)
class RunConfig:
    """Operational settings. None of these participate in the digest."""

    output_root: str = field(default="data/runs", metadata=_IMMATERIAL)
    weights_root: str = field(default="weights", metadata=_IMMATERIAL)
    workers: int = field(default=1, metadata=_IMMATERIAL)
    log_level: str = field(default="INFO", metadata=_IMMATERIAL)


@dataclass(frozen=True)
class ResolvedConfig:
    """A configuration with every value already resolved — no lookups remain."""

    species_profile: str
    species_profile_version: str
    dataset_name: str
    dataset_version: str
    models: dict[str, ModelIdentity] = field(default_factory=dict)
    seed: int = 0
    feature_set_version: str = "0"
    event_schema_version: str = "1"
    ingest: IngestConfig = field(default_factory=IngestConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    phenotype: PhenotypeConfig = field(default_factory=PhenotypeConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    run: RunConfig = field(default_factory=RunConfig)
    notes: str = field(default="", metadata=_IMMATERIAL)

    # -- digest -------------------------------------------------------------

    def material(self) -> dict[str, Any]:
        """The subset of the configuration the digest is taken over."""
        return _material_of(self)

    @property
    def digest(self) -> str:
        return hashlib.sha256(canonical_json(self.material()).encode("utf-8")).hexdigest()

    @property
    def short_digest(self) -> str:
        return self.digest[:12]

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _as_plain(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolvedConfig:
        data = dict(data)
        try:
            models = {
                role: ModelIdentity(**spec) for role, spec in (data.pop("models", {}) or {}).items()
            }
        except TypeError as exc:
            raise ConfigError(f"malformed model identity: {exc}") from exc
        sections = {
            "ingest": IngestConfig,
            "perception": PerceptionConfig,
            "identity": IdentityConfig,
            "phenotype": PhenotypeConfig,
            "baseline": BaselineConfig,
            "events": EventsConfig,
            "evaluation": EvaluationConfig,
            "run": RunConfig,
        }
        kwargs: dict[str, Any] = {"models": models}
        for name, klass in sections.items():
            kwargs[name] = klass(**(data.pop(name, {}) or {}))
        try:
            return cls(**data, **kwargs)
        except TypeError as exc:
            raise ConfigError(f"malformed configuration: {exc}") from exc

    @classmethod
    def from_yaml(cls, path: str | Path) -> ResolvedConfig:
        text = Path(path).read_text(encoding="utf-8")
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        return cls.from_dict(loaded)


def _as_plain(value: Any) -> Any:
    from dataclasses import fields as dc_fields
    from dataclasses import is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _as_plain(getattr(value, f.name)) for f in dc_fields(value)}
    if isinstance(value, dict):
        return {k: _as_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_plain(v) for v in value]
    return value


def _material_of(value: Any) -> Any:
    """Recursively keep only the fields that participate in the digest."""
    from dataclasses import fields as dc_fields
    from dataclasses import is_dataclass

    if is_dataclass(value) and not isinstance(value, type):
        out = {}
        for f in dc_fields(value):
            if f.metadata.get("material") is False:
                continue
            out[f.name] = _material_of(getattr(value, f.name))
        return out
    if isinstance(value, dict):
        return {k: _material_of(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_material_of(v) for v in value]
    return value
