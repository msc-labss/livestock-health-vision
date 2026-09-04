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
from ..errors import UndeclaredFeatureError
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
        withers = tracks["withers"].xy
        tail = tracks["base_of_tail"].xy
        usable = ~np.isnan(withers[:, 0]) & ~np.isnan(tail[:, 0])
        if usable.sum() < 3:
            return {}

        midpoint = (withers + tail) / 2.0
        body_length = np.linalg.norm(withers - tail, axis=1)
        reference_length = float(np.nanmedian(body_length[usable]))
        if not np.isfinite(reference_length) or reference_length <= 0:
            return {}

        path = midpoint[usable]
        direction = _principal_direction(path)
        lateral_axis = np.array([-direction[1], direction[0]])
        origin = path.mean(axis=0)

        progression = (midpoint - origin) @ direction
        lateral = (midpoint - origin) @ lateral_axis

        values: dict[str, float] = {}
        times = seconds[usable] if seconds is not None else None
        progression_used = progression[usable]
        lateral_used = lateral[usable]

        if times is not None and len(times) >= 2:
            deltas = np.diff(times)
            step = np.diff(progression_used)
            with np.errstate(divide="ignore", invalid="ignore"):
                velocity = np.where(deltas > 0, step / np.where(deltas > 0, deltas, np.nan), np.nan)
            velocity = velocity[np.isfinite(velocity)]
            if velocity.size:
                speed = np.abs(velocity) / reference_length
                values["speed"] = float(np.median(speed))
                mean_speed = float(np.mean(speed))
                values["speed_variability"] = (
                    float(np.std(speed) / mean_speed) if mean_speed > 0 else 0.0
                )

        values["lateral_sway"] = float(np.sqrt(np.mean(lateral_used**2)) / reference_length)
        values["tracking_jitter"] = float(_jitter(lateral_used) / reference_length)

        head = tracks["head"].xy
        head_offset = np.abs((head - origin) @ lateral_axis)
        if np.isfinite(head_offset).any():
            values["head_lateral_offset"] = float(np.nanmean(head_offset) / reference_length)

        neck = tracks["neck"].xy
        curvature = _perpendicular_distance(neck, withers, tail)
        if np.isfinite(curvature).any():
            values["spine_lateral_curvature"] = float(np.nanmean(curvature) / reference_length)

        for group, left, right in (
            ("front", "left_front_paw", "right_front_paw"),
            ("back", "left_back_paw", "right_back_paw"),
        ):
            left_signal = _relative_progression(tracks[left].xy, midpoint, direction)
            right_signal = _relative_progression(tracks[right].xy, midpoint, direction)

            left_amplitude = _excursion(left_signal)
            right_amplitude = _excursion(right_signal)
            amplitudes = [a for a in (left_amplitude, right_amplitude) if a is not None]
            if amplitudes:
                values[f"stride_length_{group}"] = float(np.mean(amplitudes) / reference_length)
            if left_amplitude is not None and right_amplitude is not None:
                total = left_amplitude + right_amplitude
                values[f"step_asymmetry_{group}"] = (
                    float(abs(left_amplitude - right_amplitude) / total) if total > 0 else 0.0
                )

            if times is not None:
                frequency = _dominant_frequency(left_signal, right_signal, seconds)
                if frequency is not None:
                    values[f"stride_frequency_{group}"] = float(frequency)

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

        usable = [f for f in features if f.quality is not QualityFlag.UNUSABLE]
        if not usable:
            return False, ValidityReason.NO_USABLE_FEATURES
        if len(usable) < self.config.phenotype.min_usable_features:
            return False, ValidityReason.TOO_FEW_USABLE_FEATURES

        # Judged over what this source could supply. A feature no pass here can
        # ever carry — a paw under a cow seen from above — says something about
        # the camera, and counting it against every pass would reject them all
        # for a fact about the mounting rather than about the animal.
        reduced = sum(1 for f in usable if f.quality is not QualityFlag.GOOD)
        if reduced / len(usable) > self.config.phenotype.max_reduced_quality_fraction:
            return False, ValidityReason.TOO_MANY_REDUCED_FEATURES
        return True, ValidityReason.VALID


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
        # Uniform resampling, so the transform's frequency axis means something.
        uniform = np.linspace(times[0], times[-1], mask.sum())
        resampled = np.interp(uniform, times, values)
        spectrum = np.abs(np.fft.rfft(resampled))
        frequencies = np.fft.rfftfreq(len(resampled), d=duration / (len(resampled) - 1))
        if len(spectrum) < 2:
            continue
        peak = int(np.argmax(spectrum[1:]) + 1)
        candidates.append(float(frequencies[peak]))
    return float(np.mean(candidates)) if candidates else None
