"""Locomotion feature extraction.

The feature set is the profile's, named and versioned. Extraction may only emit
features the profile declares; a value under any other name is an error naming
the offender rather than an extra column that quietly appears downstream.

Every feature carries a quality flag derived from the confidence and coverage of
the keypoints it was computed from, and the pass carries an overall validity
decision. Both exist so that a consumer knows what a number is worth without
having to re-derive it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import ResolvedConfig
from ..errors import (
    FeatureViewMismatchError,
    UndeclaredFeatureError,
    UnimplementedFeatureError,
)
from ..perception.schemas import Pose
from ..profiles import SpeciesProfile
from .schemas import (
    FeatureRecord,
    FeatureValue,
    LanePass,
    PassCompleteness,
    QualityFlag,
    ValidityReason,
)

__all__ = ["FeatureExtractor", "KeypointTrack"]

_AXIS = "body axis"


@dataclass
class KeypointTrack:
    """One keypoint's observed positions across a pass, with its coverage."""

    name: str
    xy: np.ndarray  # (n, 2), NaN where unobserved
    confidence: np.ndarray  # (n,)

    @property
    def coverage(self) -> float:
        if len(self.confidence) == 0:
            return 0.0
        return float(np.mean(~np.isnan(self.xy[:, 0])))

    @property
    def mean_confidence(self) -> float:
        observed = self.confidence[~np.isnan(self.xy[:, 0])]
        return float(observed.mean()) if observed.size else 0.0


class FeatureExtractor:
    """Turns the poses of one pass into a versioned feature record."""

    def __init__(self, profile: SpeciesProfile, config: ResolvedConfig) -> None:
        skeleton = profile.skeleton
        for definition in profile.feature_set.features:
            # Checked once, before any pass: a feature's view is a property of
            # its definition, so a mismatch is wrong for every pass equally and
            # there is nothing to learn by discovering it per pass.
            if definition.view and definition.view != skeleton.view:
                raise FeatureViewMismatchError(
                    feature_name=definition.name,
                    feature_view=definition.view,
                    skeleton_id=skeleton.identifier,
                    skeleton_view=skeleton.view,
                )
        self.profile = profile
        self.config = config
        self.feature_set = profile.feature_set

    # -- public -------------------------------------------------------------

    def extract(
        self,
        lane_pass: LanePass,
        poses: list[Pose],
        *,
        animal_id: str = "",
        identity_resolved: bool = True,
    ) -> FeatureRecord:
        poses = sorted(
            (
                p
                for p in poses
                if lane_pass.first_frame_index
                <= p.provenance.frame_index
                <= lane_pass.last_frame_index
            ),
            key=lambda p: p.provenance.frame_index,
        )
        tracks = self._keypoint_tracks(poses)
        seconds = self._elapsed_seconds(poses)

        computed = self._compute(tracks, seconds)
        self._reject_undeclared(computed)

        sampling_hz = _sampling_rate(seconds)
        features = tuple(
            self._as_feature_value(name, computed.get(name), tracks, sampling_hz)
            for name in self.feature_set.names
        )
        valid, reason = self._decide_validity(
            lane_pass, features, len(poses), identity_resolved=identity_resolved
        )

        return FeatureRecord(
            pass_id=lane_pass.pass_id,
            feature_set_version=self.feature_set.version,
            skeleton_id=self.profile.skeleton.identifier,
            skeleton_version=self.profile.skeleton.version,
            site_key=lane_pass.site_key,
            day_key=lane_pass.day_key,
            valid=valid,
            validity_reason=reason,
            features=features,
            tracklet_id=lane_pass.tracklet_id,
            animal_id=animal_id if identity_resolved else "",
            observed_at=lane_pass.timestamp,
            completeness=lane_pass.completeness,
        )

    # -- geometry -----------------------------------------------------------

    def _keypoint_tracks(self, poses: list[Pose]) -> dict[str, KeypointTrack]:
        names = self.profile.skeleton.names
        count = len(poses)
        tracks = {
            name: KeypointTrack(
                name=name,
                xy=np.full((count, 2), np.nan, dtype=float),
                confidence=np.zeros(count, dtype=float),
            )
            for name in names
        }
        for row, pose in enumerate(poses):
            for keypoint in pose.keypoints:
                track = tracks.get(keypoint.name)
                if track is None:
                    continue
                track.confidence[row] = keypoint.confidence
                if keypoint.observed:
                    track.xy[row] = (keypoint.x, keypoint.y)
        return tracks

    def _elapsed_seconds(self, poses: list[Pose]) -> np.ndarray | None:
        stamps = [p.provenance.capture_timestamp for p in poses]
        if not stamps or any(s is None for s in stamps):
            return None
        if not all(p.provenance.timestamp_reliable for p in poses):
            return None
        origin = stamps[0]
        return np.array([(s - origin).total_seconds() for s in stamps], dtype=float)

    def _compute(
        self, tracks: dict[str, KeypointTrack], seconds: np.ndarray | None
    ) -> dict[str, float]:
        """Run the implementation of every feature the profile declares available.

        Nothing here names a keypoint. The body axis comes from the skeleton's
        declared roles, and every other point a feature needs comes from that
        feature's own ``depends_on``. That is what lets one extractor serve two
        skeletons whose points do not even share names.
        """
        geometry = _Geometry.build(tracks, seconds, self.profile.skeleton)
        if geometry is None:
            return {}

        values: dict[str, float] = {}
        for definition in self.feature_set.available:
            implementation = _IMPLEMENTATIONS.get(definition.name)
            if implementation is None:
                # Declared available and not implemented. Refused rather than
                # skipped: silently emitting nothing would be indistinguishable
                # from a feature that legitimately could not be computed.
                raise UnimplementedFeatureError(definition.name, self.feature_set.version)
            extras = tuple(d for d in definition.depends_on if d not in geometry.axis_names)
            value = implementation(geometry, tracks, extras)
            if value is not None and np.isfinite(value):
                values[definition.name] = float(value)
        return values

    # -- declaration and quality -------------------------------------------

    def _reject_undeclared(self, computed: dict[str, float]) -> None:
        for name in computed:
            if name not in self.feature_set:
                raise UndeclaredFeatureError(name, self.feature_set.version)

    def _as_feature_value(
        self,
        name: str,
        value: float | None,
        tracks: dict[str, KeypointTrack],
        sampling_hz: float | None = None,
    ) -> FeatureValue:
        definition = self.feature_set.get(name)
        assert definition is not None  # names come from the feature set itself

        # Declared, decided, and not computable here. Recorded with its reason
        # rather than omitted, so the feature set stays legible as a decision,
        # and carried as a state of its own rather than as a bad measurement.
        if not definition.available:
            return FeatureValue(
                name=name,
                value=float("nan"),
                unit=definition.unit,
                quality=QualityFlag.UNAVAILABLE,
                coverage=0.0,
                note=definition.unavailable_reason
                or "declared by the profile but not computable under this configuration",
            )

        dependencies = [tracks[d] for d in definition.depends_on if d in tracks]
        coverages = {t.name: t.coverage for t in dependencies}
        limiting = min(coverages, key=lambda k: coverages[k]) if coverages else ""
        coverage = min(coverages.values()) if coverages else 0.0

        # A periodic feature sampled below twice its own band is not measured.
        # Reporting the number anyway would describe the frame rate.
        required = definition.requires_sampling_hz
        if required > 0 and (sampling_hz is None or sampling_hz < required):
            observed = "unknown" if sampling_hz is None else f"{sampling_hz:.3g} Hz"
            return FeatureValue(
                name=name,
                value=float("nan"),
                unit=definition.unit,
                quality=QualityFlag.UNUSABLE,
                limiting_keypoint=limiting,
                coverage=coverage,
                note=(
                    f"source sampled at {observed}; this feature needs at least "
                    f"{required:g} Hz to be resolvable"
                ),
            )

        if value is None or not np.isfinite(value):
            return FeatureValue(
                name=name,
                value=float("nan"),
                unit=definition.unit,
                quality=QualityFlag.UNUSABLE,
                limiting_keypoint=limiting,
                coverage=coverage,
                note="not computable from the observed keypoints",
            )

        confidence = min((t.mean_confidence for t in dependencies), default=0.0)
        threshold = self.config.perception.keypoint_visibility_threshold
        if coverage >= 0.8 and confidence >= threshold:
            quality, note = QualityFlag.GOOD, ""
        elif coverage >= 0.4:
            quality = QualityFlag.REDUCED
            note = f"{limiting!r} observed in {coverage:.0%} of the pass"
        else:
            quality = QualityFlag.UNUSABLE
            note = f"{limiting!r} observed in only {coverage:.0%} of the pass"

        return FeatureValue(
            name=name,
            value=float(value),
            unit=definition.unit,
            quality=quality,
            limiting_keypoint=limiting if quality is not QualityFlag.GOOD else "",
            coverage=coverage,
            note=note,
        )

    def _decide_validity(
        self,
        lane_pass: LanePass,
        features: tuple[FeatureValue, ...],
        pose_count: int,
        *,
        identity_resolved: bool,
    ) -> tuple[bool, ValidityReason]:
        if not identity_resolved:
            return False, ValidityReason.UNRESOLVED_IDENTITY
        if pose_count < self.config.phenotype.min_pass_frames:
            return False, ValidityReason.TOO_FEW_FRAMES
        if (
            lane_pass.completeness is PassCompleteness.PARTIAL
            and not self.config.phenotype.accept_partial_passes
        ):
            return False, ValidityReason.PARTIAL_PASS

        # Judged only over features this configuration can supply at all. An
        # unavailable feature is the same for every pass here, so counting it
        # would reject them all for a fact about the backend rather than about
        # the animal.
        supplied = [f for f in features if f.quality is not QualityFlag.UNAVAILABLE]
        usable = [f for f in supplied if f.quality is not QualityFlag.UNUSABLE]
        if not usable:
            return False, ValidityReason.NO_USABLE_FEATURES
        if len(usable) < self.config.phenotype.min_usable_features:
            return False, ValidityReason.TOO_FEW_USABLE_FEATURES

        reduced = sum(1 for f in usable if f.quality is not QualityFlag.GOOD)
        if reduced / len(usable) > self.config.phenotype.max_reduced_quality_fraction:
            return False, ValidityReason.TOO_MANY_REDUCED_FEATURES
        return True, ValidityReason.VALID


# -- geometry and the feature implementations -------------------------------
#
# One extractor serves every skeleton, so no implementation below names a
# keypoint. The body axis arrives through the skeleton's declared roles, and
# everything else a feature needs arrives as ``extras`` — that feature's own
# ``depends_on``, minus the axis. Which implementations run is decided by which
# features the profile declares available, never by the code.


@dataclass(frozen=True)
class _Geometry:
    """The frame of reference a pass is measured in."""

    axis_names: tuple[str, str]
    midpoint: np.ndarray
    usable: np.ndarray
    reference_length: float
    direction: np.ndarray
    # Perpendicular to travel in the image. Under a top-down view that is
    # horizontal in the world; under a lateral view it is the sagittal vertical.
    # Which physical quantity it is, is exactly what a feature's declared view
    # asserts, and what FeatureExtractor refuses to let drift.
    perpendicular: np.ndarray
    origin: np.ndarray
    seconds: np.ndarray | None

    @classmethod
    def build(cls, tracks, seconds, skeleton) -> _Geometry | None:
        cranial = skeleton.role("body_axis_cranial")
        caudal = skeleton.role("body_axis_caudal")
        head_end = tracks[cranial].xy
        tail_end = tracks[caudal].xy
        usable = ~np.isnan(head_end[:, 0]) & ~np.isnan(tail_end[:, 0])
        if usable.sum() < 3:
            return None

        midpoint = (head_end + tail_end) / 2.0
        lengths = np.linalg.norm(head_end - tail_end, axis=1)
        reference_length = float(np.nanmedian(lengths[usable]))
        if not np.isfinite(reference_length) or reference_length <= 0:
            return None

        path = midpoint[usable]
        direction = _principal_direction(path)
        return cls(
            axis_names=(cranial, caudal),
            midpoint=midpoint,
            usable=usable,
            reference_length=reference_length,
            direction=direction,
            perpendicular=np.array([-direction[1], direction[0]]),
            origin=path.mean(axis=0),
            seconds=seconds,
        )

    @property
    def times(self) -> np.ndarray | None:
        return self.seconds[self.usable] if self.seconds is not None else None

    def along(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.direction

    def across(self, points: np.ndarray) -> np.ndarray:
        return (points - self.origin) @ self.perpendicular

    def speeds(self) -> np.ndarray | None:
        times = self.times
        if times is None or len(times) < 2:
            return None
        progression = self.along(self.midpoint)[self.usable]
        deltas = np.diff(times)
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.where(
                deltas > 0, np.diff(progression) / np.where(deltas > 0, deltas, np.nan), np.nan
            )
        velocity = velocity[np.isfinite(velocity)]
        return np.abs(velocity) / self.reference_length if velocity.size else None


def _f_speed(geo, tracks, extras):
    speeds = geo.speeds()
    return float(np.median(speeds)) if speeds is not None else None


def _f_speed_variability(geo, tracks, extras):
    speeds = geo.speeds()
    if speeds is None:
        return None
    mean = float(np.mean(speeds))
    return float(np.std(speeds) / mean) if mean > 0 else 0.0


def _f_sway(geo, tracks, extras):
    """RMS excursion of the body across its own path of travel."""
    across = geo.across(geo.midpoint)[geo.usable]
    return float(np.sqrt(np.mean(across**2)) / geo.reference_length)


def _f_tracking_jitter(geo, tracks, extras):
    across = geo.across(geo.midpoint)[geo.usable]
    return float(_jitter(across) / geo.reference_length)


def _f_point_offset_across(geo, tracks, extras):
    """Mean absolute offset of a single point from the axis, across travel."""
    if not extras:
        return None
    offset = np.abs(geo.across(tracks[extras[0]].xy))
    return float(np.nanmean(offset) / geo.reference_length) if np.isfinite(offset).any() else None


def _f_point_oscillation_across(geo, tracks, extras):
    """Excursion of a point across travel, with the body's own motion removed.

    Under a lateral view this is vertical head oscillation — head bob. Removing
    the body's component keeps a camera that is not perfectly level from reading
    as movement of the animal.
    """
    if not extras:
        return None
    body = geo.across(geo.midpoint)
    signal = geo.across(tracks[extras[0]].xy) - body
    excursion = _excursion(signal)
    return float(excursion / geo.reference_length) if excursion is not None else None


def _f_curvature(geo, tracks, extras):
    """Deviation of a point from the straight line joining the axis ends."""
    if not extras:
        return None
    cranial, caudal = geo.axis_names
    distance = _perpendicular_distance(tracks[extras[0]].xy, tracks[cranial].xy, tracks[caudal].xy)
    return (
        float(np.nanmedian(distance) / geo.reference_length)
        if np.isfinite(distance).any()
        else None
    )


def _distal_excursions(geo, tracks, extras):
    out = []
    for name in extras:
        signal = _relative_progression(tracks[name].xy, geo.midpoint, geo.direction)
        amplitude = _excursion(signal)
        if amplitude is not None:
            out.append(amplitude)
    return out


def _f_stride_length(geo, tracks, extras):
    amplitudes = _distal_excursions(geo, tracks, extras)
    return float(np.mean(amplitudes) / geo.reference_length) if amplitudes else None


def _f_step_asymmetry(geo, tracks, extras):
    amplitudes = _distal_excursions(geo, tracks, extras)
    if len(amplitudes) != 2:
        return None
    total = amplitudes[0] + amplitudes[1]
    return float(abs(amplitudes[0] - amplitudes[1]) / total) if total > 0 else 0.0


def _f_stride_frequency(geo, tracks, extras):
    if len(extras) != 2 or geo.seconds is None:
        return None
    left = _relative_progression(tracks[extras[0]].xy, geo.midpoint, geo.direction)
    right = _relative_progression(tracks[extras[1]].xy, geo.midpoint, geo.direction)
    return _dominant_frequency(left, right, geo.seconds)


# Feature name -> implementation. A profile decides which of these run by
# declaring the feature available; a declared feature missing from this table is
# refused rather than quietly skipped.
_IMPLEMENTATIONS = {
    # feature-set version 1, lateral
    "speed": _f_speed,
    "stride_length": _f_stride_length,
    "head_bob": _f_point_oscillation_across,
    # feature-set version 0, top-down
    "speed_variability": _f_speed_variability,
    "lateral_sway": _f_sway,
    "tracking_jitter": _f_tracking_jitter,
    "head_lateral_offset": _f_point_offset_across,
    "spine_lateral_curvature": _f_curvature,
    "stride_length_front": _f_stride_length,
    "stride_length_back": _f_stride_length,
    "step_asymmetry_front": _f_step_asymmetry,
    "step_asymmetry_back": _f_step_asymmetry,
    "stride_frequency_front": _f_stride_frequency,
    "stride_frequency_back": _f_stride_frequency,
}


# -- numerics ---------------------------------------------------------------


def _sampling_rate(seconds: np.ndarray | None) -> float | None:
    """Effective frames per second of a pass, or None when it has no clock."""
    if seconds is None or len(seconds) < 2:
        return None
    span = float(seconds[-1] - seconds[0])
    if span <= 0:
        return None
    return (len(seconds) - 1) / span


def _principal_direction(path: np.ndarray) -> np.ndarray:
    """The unit direction of travel, from the principal axis of the path."""
    centred = path - path.mean(axis=0)
    if len(centred) < 2 or not np.isfinite(centred).all():
        return np.array([0.0, 1.0])
    _, _, vectors = np.linalg.svd(centred, full_matrices=False)
    direction = vectors[0]
    # Orient it along actual travel, so progression increases with time.
    if float((path[-1] - path[0]) @ direction) < 0:
        direction = -direction
    norm = np.linalg.norm(direction)
    return direction / norm if norm else np.array([0.0, 1.0])


def _relative_progression(
    keypoint: np.ndarray, midpoint: np.ndarray, direction: np.ndarray
) -> np.ndarray:
    """A paw's along-axis position relative to the body, which is what swings."""
    return (keypoint - midpoint) @ direction


def _excursion(signal: np.ndarray) -> float | None:
    finite = signal[np.isfinite(signal)]
    if finite.size < 3:
        return None
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))


def _jitter(signal: np.ndarray) -> float:
    if len(signal) < 3:
        return 0.0
    smoothed = np.convolve(signal, np.ones(3) / 3.0, mode="same")
    return float(np.median(np.abs(signal - smoothed)))


def _perpendicular_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of ``point`` from the line through ``a`` and ``b``, per row."""
    line = b - a
    lengths = np.linalg.norm(line, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = np.abs(line[:, 0] * (a[:, 1] - point[:, 1]) - (a[:, 0] - point[:, 0]) * line[:, 1])
        return np.where(lengths > 0, cross / np.where(lengths > 0, lengths, np.nan), np.nan)


def _dominant_frequency(
    left: np.ndarray, right: np.ndarray, seconds: np.ndarray | None
) -> float | None:
    """Dominant oscillation frequency of the limb pair, in cycles per second."""
    if seconds is None:
        return None
    candidates = []
    for signal in (left, right):
        mask = np.isfinite(signal) & np.isfinite(seconds)
        if mask.sum() < 6:
            continue
        times = seconds[mask]
        values = signal[mask] - np.mean(signal[mask])
        duration = float(times[-1] - times[0])
        if duration <= 0:
            continue
        uniform = np.linspace(times[0], times[-1], mask.sum())
        resampled = np.interp(uniform, times, values)
        spectrum = np.abs(np.fft.rfft(resampled))
        frequencies = np.fft.rfftfreq(len(resampled), d=duration / (len(resampled) - 1))
        if len(spectrum) < 2:
            continue
        peak = int(np.argmax(spectrum[1:]) + 1)
        candidates.append(float(frequencies[peak]))
    return float(np.mean(candidates)) if candidates else None
