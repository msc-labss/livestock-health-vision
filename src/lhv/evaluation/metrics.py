"""The three metric families, kept apart.

Perception, phenotype and operational metrics answer different questions and
fail for different reasons. Averaging them into a headline number would let a
strong detector hide a useless alarm rate, which is precisely the failure mode
the literature keeps reproducing. The harness therefore has no way to combine
them.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..perception.schemas import BoundingBox, Pose, Tracklet, Visibility

__all__ = [
    "LabelledBox",
    "LabelledKeypoint",
    "MetricFamily",
    "detection_metrics",
    "tracking_metrics",
    "pose_metrics",
    "phenotype_metrics",
    "identity_metrics",
    "operational_metrics",
]


def _frame_key(detection) -> tuple[str, int]:
    return (detection.provenance.source_id, detection.provenance.frame_index)


@dataclass(frozen=True)
class LabelledBox:
    """Ground truth for one animal in one frame of one source.

    ``source_id`` is part of the key, not decoration. Frame indices restart at
    zero in every source, so matching on the frame index alone would score a
    prediction from one sequence against the labels of another.
    """

    frame_index: int
    box: BoundingBox
    track_id: str = ""
    animal_id: str = ""
    source_id: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.source_id, self.frame_index)


@dataclass(frozen=True)
class LabelledKeypoint:
    frame_index: int
    name: str
    x: float
    y: float
    visible: bool = True
    track_id: str = ""
    source_id: str = ""

    @property
    def key(self) -> tuple[str, int, str]:
        return (self.source_id, self.frame_index, self.name)


@dataclass
class MetricFamily:
    """One family of metrics, with the notes that make them readable."""

    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        lines = [f"{self.name}:"]
        for key, value in sorted(self.metrics.items()):
            lines.append(f"  {key}: {value:.4f}")
        for key, value in sorted(self.counts.items()):
            lines.append(f"  {key}: {value}")
        for note in self.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines)


# -- perception -------------------------------------------------------------


def detection_metrics(
    predictions: Iterable, labels: Iterable[LabelledBox], *, iou_threshold: float = 0.5
) -> MetricFamily:
    """Greedy one-to-one matching by overlap, per frame."""
    predicted_by_frame: dict[tuple[str, int], list] = {}
    for detection in predictions:
        predicted_by_frame.setdefault(_frame_key(detection), []).append(detection)
    labels_by_frame: dict[tuple[str, int], list[LabelledBox]] = {}
    for label in labels:
        labels_by_frame.setdefault(label.key, []).append(label)

    true_positives = false_positives = false_negatives = 0
    matched_ious: list[float] = []

    for frame in sorted(set(predicted_by_frame) | set(labels_by_frame)):
        predicted = predicted_by_frame.get(frame, [])
        truth = labels_by_frame.get(frame, [])
        pairs = sorted(
            (
                (p.box.iou(t.box), pi, ti)
                for pi, p in enumerate(predicted)
                for ti, t in enumerate(truth)
            ),
            key=lambda item: (-item[0], item[1], item[2]),
        )
        used_predictions: set[int] = set()
        used_truth: set[int] = set()
        for score, pi, ti in pairs:
            if score < iou_threshold or pi in used_predictions or ti in used_truth:
                continue
            used_predictions.add(pi)
            used_truth.add(ti)
            matched_ious.append(score)
        true_positives += len(used_predictions)
        false_positives += len(predicted) - len(used_predictions)
        false_negatives += len(truth) - len(used_truth)

    precision = true_positives / (true_positives + false_positives) if predicted_by_frame else 0.0
    recall = true_positives / (true_positives + false_negatives) if labels_by_frame else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return MetricFamily(
        name="detection",
        metrics={
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": statistics.fmean(matched_ious) if matched_ious else 0.0,
        },
        counts={
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        notes=[f"matching threshold: IoU >= {iou_threshold}"],
    )


def tracking_metrics(
    tracklets: Sequence[Tracklet],
    labels: Iterable[LabelledBox],
    *,
    iou_threshold: float = 0.5,
) -> MetricFamily:
    """Identity continuity, measured against labelled track identifiers."""
    labels = list(labels)
    labels_by_frame: dict[tuple[str, int], list[LabelledBox]] = {}
    for label in labels:
        labels_by_frame.setdefault(label.key, []).append(label)

    # For each predicted tracklet, which labelled track it matched frame by frame.
    assignments: dict[str, list[str]] = {}
    for tracklet in tracklets:
        sequence: list[str] = []
        for detection in tracklet.detections:
            best, best_score = "", 0.0
            for label in labels_by_frame.get(_frame_key(detection), []):
                score = detection.box.iou(label.box)
                if score > best_score:
                    best, best_score = label.track_id, score
            if best_score >= iou_threshold and best:
                sequence.append(best)
        assignments[tracklet.tracklet_id] = sequence

    switches = 0
    purities: list[float] = []
    for sequence in assignments.values():
        if not sequence:
            continue
        switches += sum(1 for a, b in zip(sequence, sequence[1:], strict=False) if a != b)
        dominant = max(set(sequence), key=sequence.count)
        purities.append(sequence.count(dominant) / len(sequence))

    truth_lengths: dict[str, int] = {}
    for label in labels:
        truth_lengths[label.track_id] = truth_lengths.get(label.track_id, 0) + 1
    covered: dict[str, int] = {}
    for sequence in assignments.values():
        for track_id in set(sequence):
            covered[track_id] = max(covered.get(track_id, 0), sequence.count(track_id))
    mostly_tracked = sum(
        1
        for track_id, length in truth_lengths.items()
        if length and covered.get(track_id, 0) / length >= 0.8
    )
    fragments: dict[str, int] = {}
    for sequence in assignments.values():
        for track_id in set(sequence):
            fragments[track_id] = fragments.get(track_id, 0) + 1

    return MetricFamily(
        name="tracking",
        metrics={
            "mean_tracklet_purity": statistics.fmean(purities) if purities else 0.0,
            "mostly_tracked_fraction": (
                mostly_tracked / len(truth_lengths) if truth_lengths else 0.0
            ),
            "mean_fragmentation": (statistics.fmean(fragments.values()) if fragments else 0.0),
        },
        counts={
            "identity_switches": switches,
            "predicted_tracklets": len(tracklets),
            "labelled_tracks": len(truth_lengths),
        },
        notes=["a labelled track is mostly tracked when >= 80% of its frames are covered"],
    )


def pose_metrics(
    poses: Iterable[Pose],
    labels: Iterable[LabelledKeypoint],
    *,
    distance_threshold: float = 0.1,
    normaliser: float = 100.0,
) -> MetricFamily:
    """Percentage of correct keypoints, plus how honestly visibility was reported."""
    poses = list(poses)
    truth: dict[tuple[str, int, str], LabelledKeypoint] = {label.key: label for label in labels}
    correct = evaluated = comparable = 0
    claimed_visible = truly_visible = agreed = 0
    invisible_but_claimed = 0

    for pose in poses:
        for keypoint in pose.keypoints:
            label = truth.get(
                (pose.provenance.source_id, pose.provenance.frame_index, keypoint.name)
            )
            if label is None:
                continue
            comparable += 1
            observed = keypoint.visibility is Visibility.VISIBLE
            claimed_visible += int(observed)
            truly_visible += int(label.visible)
            agreed += int(observed == label.visible)
            if observed and not label.visible:
                invisible_but_claimed += 1
            if not label.visible or not observed:
                continue
            evaluated += 1
            distance = ((keypoint.x - label.x) ** 2 + (keypoint.y - label.y) ** 2) ** 0.5
            if distance / normaliser <= distance_threshold:
                correct += 1

    return MetricFamily(
        name="pose",
        metrics={
            "pck": correct / evaluated if evaluated else 0.0,
            "visibility_agreement": agreed / comparable if comparable else 0.0,
        },
        counts={
            "keypoints_compared": comparable,
            "keypoints_evaluated": evaluated,
            "keypoints_correct": correct,
            "claimed_visible": claimed_visible,
            "labelled_visible": truly_visible,
            "claimed_visible_but_occluded": invisible_but_claimed,
        },
        notes=[
            f"a keypoint is correct within {distance_threshold:.0%} of the normalising length "
            f"({normaliser:g} px)",
            "only keypoints labelled visible and emitted visible are scored for position",
        ],
    )


# -- phenotype --------------------------------------------------------------


def phenotype_metrics(observations: Sequence, feature_names: Sequence[str]) -> MetricFamily:
    """Reproducibility and stability of features across repeated passes.

    No clinical label appears anywhere in this computation, and none is
    required. What is measured is whether the same animal, measured twice,
    yields the same number — which is a precondition for the feature meaning
    anything, and is not the same thing as the feature meaning something.
    """
    by_animal: dict[str, list] = {}
    for observation in observations:
        by_animal.setdefault(observation.animal_id, []).append(observation)

    metrics: dict[str, float] = {}
    repeated = {a: obs for a, obs in by_animal.items() if len(obs) >= 2}

    for name in feature_names:
        within: list[float] = []
        animal_means: list[float] = []
        for values in (
            [o.features[name] for o in obs if o.features.get(name) is not None]
            for obs in repeated.values()
        ):
            if len(values) < 2:
                continue
            mean = statistics.fmean(values)
            animal_means.append(mean)
            if mean != 0:
                within.append(statistics.stdev(values) / abs(mean))
        if within:
            metrics[f"{name}.within_animal_cv"] = statistics.fmean(within)
        if len(animal_means) >= 2:
            between = statistics.stdev(animal_means)
            metrics[f"{name}.between_animal_sd"] = between
            within_sd = statistics.fmean(
                [
                    statistics.stdev([o.features[name] for o in obs if name in o.features])
                    for obs in repeated.values()
                    if len([o for o in obs if name in o.features]) >= 2
                ]
                or [0.0]
            )
            total = between**2 + within_sd**2
            # Intraclass correlation: how much of the variance is between
            # animals rather than within one. A feature that cannot tell two
            # animals apart cannot tell one animal's two days apart either.
            metrics[f"{name}.icc"] = (between**2 / total) if total > 0 else 0.0

    return MetricFamily(
        name="phenotype",
        metrics=metrics,
        counts={
            "animals": len(by_animal),
            "animals_with_repeated_passes": len(repeated),
            "observations": len(observations),
        },
        notes=[
            "computed without any clinical label; measures reproducibility, not validity",
            "clinical correlation is unevaluated in P0 and no number here stands in for it",
        ],
    )


# -- identity ---------------------------------------------------------------


def identity_metrics(assignments: Iterable, truth: dict[str, str], *, method: str) -> MetricFamily:
    """One assignment method's standalone accuracy against labelled identity."""
    considered = [a for a in assignments if str(a.method) == method]
    correct = sum(1 for a in considered if truth.get(a.tracklet_id) == a.animal_id)
    resolved = [a for a in considered if a.resolved]
    return MetricFamily(
        name=f"identity:{method}",
        metrics={
            "accuracy": (
                sum(1 for a in resolved if truth.get(a.tracklet_id) == a.animal_id) / len(resolved)
                if resolved
                else 0.0
            ),
            "resolution_rate": len(resolved) / len(considered) if considered else 0.0,
        },
        counts={
            "assignments": len(considered),
            "resolved": len(resolved),
            "correct": correct,
        },
        notes=[f"reported for method {method!r} alone, not pooled with any other path"],
    )


# -- operational ------------------------------------------------------------


def operational_metrics(
    events: Sequence,
    *,
    animal_days: float,
    policy_identity: str,
    reference_event_name: str,
    reference_times: dict[str, object] | None = None,
) -> MetricFamily:
    """Alarm burden and lead time — the endpoints a farmer actually feels."""
    from .report import EventLevelName

    alerts = [e for e in events if str(e.level) == EventLevelName.ALERT]
    burden = (len(alerts) / animal_days * 1000.0) if animal_days > 0 else 0.0

    metrics = {"alarm_burden_per_1000_animal_days": burden}
    counts = {
        "alerts": len(alerts),
        "events": len(events),
        "animal_days": int(animal_days),
    }
    notes = [
        f"alarm burden is produced by threshold policy {policy_identity}",
        f"lead time is measured against the reference event: {reference_event_name}",
    ]

    if reference_times:
        lead_times = []
        first_alert: dict[str, object] = {}
        for event in sorted(alerts, key=lambda e: e.event_timestamp):
            first_alert.setdefault(event.animal_id, event.event_timestamp)
        for animal_id, reference in reference_times.items():
            alert_at = first_alert.get(animal_id)
            if alert_at is None:
                continue
            lead_times.append((reference - alert_at).total_seconds() / 86400.0)
        if lead_times:
            metrics["median_lead_time_days"] = statistics.median(lead_times)
            counts["animals_with_a_reference_event"] = len(lead_times)
    else:
        notes.append(
            "no reference event was supplied, so lead time is not reported rather than "
            "being reported as zero"
        )

    return MetricFamily(name="operational", metrics=metrics, counts=counts, notes=notes)
