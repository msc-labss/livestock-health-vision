"""The species profile — the single seam where species knowledge lives.

Everything species-specific is carried here: the skeleton definition and its
version, the pretrained weight references with their licences, the feature-set
definition and version, normal-range priors and the scoring scale. Adding a
species must touch this object and nothing else; the check in
``tools/check_species_seam.py`` fails the build if another module names one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..errors import ProfileError

__all__ = [
    "KeypointDefinition",
    "SkeletonDefinition",
    "WeightReference",
    "FeatureDefinition",
    "FeatureSet",
    "NormalRange",
    "MeasuredAnchor",
    "ScoringScale",
    "SpeciesProfile",
    "load_profile",
    "available_profiles",
    "PROFILE_DIR",
]

PROFILE_DIR = Path(__file__).parent / "definitions"


@dataclass(frozen=True)
class KeypointDefinition:
    name: str
    index: int
    description: str = ""
    # The releasing dataset's own name for this keypoint, so a label file can be
    # read by name as well as by index.
    alias: str = ""
    # The mirrored keypoint, for left/right flips.
    swap: str = ""
    # Object-keypoint-similarity scale, as published with the skeleton.
    sigma: float = 0.0


@dataclass(frozen=True)
class SkeletonDefinition:
    """A named, versioned keypoint convention plus the links between keypoints."""

    identifier: str
    version: str
    view: str
    keypoints: tuple[KeypointDefinition, ...]
    links: tuple[tuple[str, str], ...] = ()
    source: str = ""
    # True when the point set is not taken from a published convention and is
    # expected to be revised. Recorded so a consumer can say so rather than
    # presenting a working guess as settled.
    provisional: bool = False
    # Named structural roles, so a stage can ask for "the body axis" instead of
    # naming keypoints that differ between skeletons.
    roles: dict[str, str] = field(default_factory=dict)
    # Where the OKS sigmas came from, or "none-established" when no published
    # sigma exists for this point set. A sigma with no provenance would put an
    # unfounded number into every keypoint metric.
    sigma_source: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(k.name for k in self.keypoints)

    @property
    def aliases(self) -> dict[str, str]:
        """The release's own keypoint names, mapped to this profile's names."""
        return {k.alias: k.name for k in self.keypoints if k.alias}

    def role(self, name: str) -> str:
        """The keypoint filling a named structural role."""
        try:
            return self.roles[name]
        except KeyError:
            raise ProfileError(
                f"skeleton {self.identifier}@{self.version} declares no role {name!r}; "
                f"extraction needs it and will not guess a keypoint"
            ) from None

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(k.sigma for k in sorted(self.keypoints, key=lambda k: k.index))

    def by_index(self) -> tuple[KeypointDefinition, ...]:
        return tuple(sorted(self.keypoints, key=lambda k: k.index))

    def index_of(self, name: str) -> int:
        for keypoint in self.keypoints:
            if keypoint.name == name:
                return keypoint.index
        raise ProfileError(f"skeleton {self.identifier}@{self.version} has no keypoint {name!r}")

    def __len__(self) -> int:
        return len(self.keypoints)


@dataclass(frozen=True)
class WeightReference:
    """A pretrained weight file, its licence and its access terms.

    Licences live here because resolving them is a profile edit, never a code
    change. So does everything a stage would otherwise have to know about the
    animal: which classes the detector should keep, and how the weights' native
    keypoint convention maps onto the profile skeleton.
    """

    role: str
    name: str
    version: str
    uri: str
    licence: str
    licence_url: str = ""
    commercial_use: str = "unknown"
    sha256: str = ""
    # Weights not trained for this task, species, or skeleton, wired in so a
    # contract can be exercised. Declared here as a field rather than described
    # in ``notes`` because prose is not reachable by a report, and output from
    # such weights must be identifiable wherever it lands.
    placeholder: bool = False
    placeholder_reason: str = ""
    notes: str = ""
    target_classes: tuple[str, ...] = ()
    native_skeleton: str = ""
    keypoint_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FeatureDefinition:
    """One locomotion feature: its name, unit, and the keypoints it depends on."""

    name: str
    unit: str
    description: str
    depends_on: tuple[str, ...]
    higher_is_worse: bool = True
    # The skeleton view under which this definition holds. A feature computed in
    # image coordinates measures a different physical quantity under a different
    # camera geometry while its name, unit and priors stay the same, so the view
    # travels with the definition and extraction refuses a mismatch.
    view: str = ""
    # False when the feature is decided and declared but cannot be computed
    # under the current skeleton or backend. Declared-and-unavailable keeps the
    # decision visible and keeps the reason recorded; omitting it would leave a
    # reader unable to tell it from a feature nobody wanted.
    available: bool = True
    unavailable_reason: str = ""
    # Frames per second the source must supply for this feature to mean
    # anything. A periodic quantity sampled below twice its own band is not
    # measured badly, it is not measured at all, and the number that comes back
    # is an artefact of the sampling rather than of the animal.
    requires_sampling_hz: float = 0.0


@dataclass(frozen=True)
class FeatureSet:
    version: str
    features: tuple[FeatureDefinition, ...]
    rationale: str = ""

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features)

    @property
    def available(self) -> tuple[FeatureDefinition, ...]:
        """Features the current skeleton and backend can actually compute."""
        return tuple(f for f in self.features if f.available)

    @property
    def unavailable(self) -> tuple[FeatureDefinition, ...]:
        """Declared features that cannot be computed, each carrying its reason."""
        return tuple(f for f in self.features if not f.available)

    def get(self, name: str) -> FeatureDefinition | None:
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def __contains__(self, name: object) -> bool:
        return name in self.names

    def __len__(self) -> int:
        return len(self.features)


@dataclass(frozen=True)
class NormalRange:
    """A prior on a feature's normal range, used only where no history exists."""

    feature: str
    low: float
    high: float
    source: str = "prior"


@dataclass(frozen=True)
class MeasuredAnchor:
    """A published central value for a feature, with the way lameness moves it.

    Deliberately not a :class:`NormalRange`. The published figures behind these
    are group means with standard errors, which say where a group sat, not how
    far a population spreads; recorded as a band they would reject most normal
    animals. An anchor informs a direction and cannot become a threshold.
    """

    feature: str
    healthy: float
    unit: str
    direction: str
    source: str
    lame_observed: float | None = None
    # True where the unit is absolute and this project has no camera
    # calibration, so the value cannot be compared against a measurement.
    calibration_required: bool = False


@dataclass(frozen=True)
class ScoringScale:
    """The clinical scale this species is conventionally scored on."""

    name: str
    minimum: float
    maximum: float
    step: float
    reference: str = ""


@dataclass(frozen=True)
class SpeciesProfile:
    species: str
    version: str
    skeleton: SkeletonDefinition
    weights: dict[str, WeightReference]
    feature_set: FeatureSet
    priors: dict[str, NormalRange]
    anchors: dict[str, MeasuredAnchor] = field(default_factory=dict)
    scoring_scale: ScoringScale | None = None
    notes: str = ""

    @property
    def identifier(self) -> str:
        return f"{self.species}@{self.version}"

    def weight(self, role: str) -> WeightReference:
        try:
            return self.weights[role]
        except KeyError as exc:
            raise ProfileError(
                f"profile {self.identifier} declares no weights for role {role!r}"
            ) from exc

    def prior(self, feature: str) -> NormalRange | None:
        return self.priors.get(feature)

    def anchor(self, feature: str) -> MeasuredAnchor | None:
        return self.anchors.get(feature)

    def unlicensed_weights(self) -> tuple[WeightReference, ...]:
        """Weight references whose licence is unresolved. Reported, never assumed away."""
        return tuple(w for w in self.weights.values() if w.licence.lower() in {"", "unknown"})

    def placeholder_weights(self) -> tuple[WeightReference, ...]:
        """Weight references declared as placeholders. Reported wherever they are used."""
        return tuple(w for w in self.weights.values() if w.placeholder)


# -- loading ----------------------------------------------------------------


def _require(data: dict[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise ProfileError(f"{where}: missing required key {key!r}")
    return data[key]


def profile_from_dict(data: dict[str, Any], *, where: str = "<profile>") -> SpeciesProfile:
    species = _require(data, "species", where)
    version = str(_require(data, "version", where))

    raw_skeleton = _require(data, "skeleton", where)
    keypoints = tuple(
        KeypointDefinition(
            name=k["name"],
            index=int(k.get("index", i)),
            description=k.get("description", ""),
            alias=k.get("alias", ""),
            swap=k.get("swap", ""),
            sigma=float(k.get("sigma", 0.0)),
        )
        for i, k in enumerate(_require(raw_skeleton, "keypoints", f"{where}.skeleton"))
    )
    skeleton = SkeletonDefinition(
        identifier=_require(raw_skeleton, "identifier", f"{where}.skeleton"),
        version=str(_require(raw_skeleton, "version", f"{where}.skeleton")),
        view=raw_skeleton.get("view", "unspecified"),
        provisional=bool(raw_skeleton.get("provisional", False)),
        roles=dict(raw_skeleton.get("roles", {}) or {}),
        sigma_source=str(raw_skeleton.get("sigma_source", "")),
        keypoints=keypoints,
        links=tuple((a, b) for a, b in raw_skeleton.get("links", [])),
        source=raw_skeleton.get("source", ""),
    )

    weights = {}
    for role, spec in (_require(data, "weights", where) or {}).items():
        weights[role] = WeightReference(
            role=role,
            name=_require(spec, "name", f"{where}.weights.{role}"),
            version=str(_require(spec, "version", f"{where}.weights.{role}")),
            uri=spec.get("uri", ""),
            licence=_require(spec, "licence", f"{where}.weights.{role}"),
            licence_url=spec.get("licence_url", ""),
            commercial_use=str(spec.get("commercial_use", "unknown")),
            sha256=spec.get("sha256", ""),
            placeholder=bool(spec.get("placeholder", False)),
            placeholder_reason=str(spec.get("placeholder_reason", "")),
            notes=spec.get("notes", ""),
            target_classes=tuple(spec.get("target_classes", ()) or ()),
            native_skeleton=spec.get("native_skeleton", ""),
            keypoint_map=dict(spec.get("keypoint_map", {}) or {}),
        )

    raw_features = _require(data, "feature_set", where)
    feature_set = FeatureSet(
        version=str(_require(raw_features, "version", f"{where}.feature_set")),
        features=tuple(
            FeatureDefinition(
                name=_require(f, "name", f"{where}.feature_set"),
                unit=_require(f, "unit", f"{where}.feature_set"),
                description=f.get("description", ""),
                depends_on=tuple(f.get("depends_on", [])),
                higher_is_worse=bool(f.get("higher_is_worse", True)),
                view=str(f.get("view", "")),
                available=bool(f.get("available", True)),
                unavailable_reason=str(f.get("unavailable_reason", "")),
                requires_sampling_hz=float(f.get("requires_sampling_hz", 0.0)),
            )
            for f in _require(raw_features, "features", f"{where}.feature_set")
        ),
        rationale=raw_features.get("rationale", ""),
    )

    priors = {
        name: NormalRange(
            feature=name,
            low=float(spec["low"]),
            high=float(spec["high"]),
            source=spec.get("source", "prior"),
        )
        for name, spec in (data.get("priors") or {}).items()
    }

    anchors = {
        name: MeasuredAnchor(
            feature=name,
            healthy=float(spec["healthy"]),
            unit=str(spec.get("unit", "")),
            direction=str(spec.get("direction", "")),
            source=str(_require(spec, "source", f"{where}.anchors.{name}")),
            lame_observed=(
                float(spec["lame_observed"]) if spec.get("lame_observed") is not None else None
            ),
            calibration_required=bool(spec.get("calibration_required", False)),
        )
        for name, spec in (data.get("anchors") or {}).items()
    }

    scale_spec = data.get("scoring_scale")
    scoring_scale = (
        ScoringScale(
            name=_require(scale_spec, "name", f"{where}.scoring_scale"),
            minimum=float(_require(scale_spec, "minimum", f"{where}.scoring_scale")),
            maximum=float(_require(scale_spec, "maximum", f"{where}.scoring_scale")),
            step=float(scale_spec.get("step", 1.0)),
            reference=scale_spec.get("reference", ""),
        )
        if scale_spec
        else None
    )

    unknown_dependencies = {
        dependency
        for feature in feature_set.features
        for dependency in feature.depends_on
        if dependency not in skeleton.names
    }
    if unknown_dependencies:
        raise ProfileError(
            f"{where}: feature set {feature_set.version} depends on keypoints absent from "
            f"skeleton {skeleton.identifier}@{skeleton.version}: "
            f"{', '.join(sorted(unknown_dependencies))}"
        )

    return SpeciesProfile(
        species=species,
        version=version,
        skeleton=skeleton,
        weights=weights,
        feature_set=feature_set,
        priors=priors,
        anchors=anchors,
        scoring_scale=scoring_scale,
        notes=data.get("notes", ""),
    )


def load_profile(name: str, *, directory: Path | None = None) -> SpeciesProfile:
    """Load a profile by name from the profile directory."""
    directory = directory or PROFILE_DIR
    path = directory / f"{name}.yaml"
    if not path.exists():
        available = ", ".join(available_profiles(directory=directory)) or "none"
        raise ProfileError(f"no species profile named {name!r} (available: {available})")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return profile_from_dict(data, where=str(path))


def available_profiles(*, directory: Path | None = None) -> tuple[str, ...]:
    directory = directory or PROFILE_DIR
    if not directory.exists():
        return ()
    return tuple(sorted(p.stem for p in directory.glob("*.yaml")))


def default_profile_name(*, directory: Path | None = None) -> str:
    """The profile to use when none was named.

    Resolved from what is installed rather than written down anywhere outside
    this package, so no caller has to name a species in order to have a default.
    """
    names = available_profiles(directory=directory)
    if not names:
        raise ProfileError("no species profile is available")
    if len(names) == 1:
        return names[0]

    # More than one is installed, so "the only one present" no longer identifies
    # the default and a profile has to say so itself. Read from the file rather
    # than the parsed profile: resolving a default must not depend on every
    # other profile being loadable.
    directory = directory or PROFILE_DIR
    declared = [
        name
        for name in names
        if (yaml.safe_load((directory / f"{name}.yaml").read_text(encoding="utf-8")) or {}).get(
            "default", False
        )
    ]
    if len(declared) == 1:
        return declared[0]
    if not declared:
        raise ProfileError(
            f"{len(names)} species profiles are available ({', '.join(names)}) and none declares "
            f"itself the default; name one explicitly or add 'default: true' to one"
        )
    raise ProfileError(
        f"more than one species profile declares itself the default "
        f"({', '.join(declared)}); exactly one may"
    )
