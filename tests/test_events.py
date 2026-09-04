"""Events: schema, threshold policy, evidence retention, masking, export."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from lhv.baseline import AssessmentState, BaselineWindow, RiskAssessment
from lhv.baseline.schemas import BaselineKind
from lhv.errors import MissingFieldError
from lhv.events import (
    ClipRetainer,
    ConsumerUnavailable,
    EventBuilder,
    EventExporter,
    EventLevel,
    FileExportAdapter,
    HealthEvent,
    MaskingError,
    ObservationWindow,
    PolicyIdentity,
    RegionMasker,
    ThresholdPolicy,
)
from lhv.profiles import load_profile

from .conftest import feature_record, write_video

BASE = datetime(2024, 3, 1, 6, 30, tzinfo=UTC)
WINDOW = ObservationWindow(start=BASE, end=BASE + timedelta(seconds=6))


@pytest.fixture
def profile():
    return load_profile("cattle")


def _assessment(*, risk: float | None = 4.0, state=AssessmentState.SCORED, injected=False, **kw):
    own = BaselineWindow(
        kind=BaselineKind.OWN_HISTORY,
        start=BASE - timedelta(days=21),
        end=BASE,
        lookback_days=21,
        animal_id="a1",
        observation_count=9,
    )
    herd = BaselineWindow(
        kind=BaselineKind.HERD,
        start=BASE - timedelta(days=3),
        end=BASE,
        lookback_days=3,
        site_key="site-a",
        observation_count=30,
        animal_count=6,
    )
    base = dict(
        assessment_id="p1:risk",
        animal_id="a1",
        observed_at=BASE,
        site_key="site-a",
        day_key="2024-03-01",
        state=state,
        scale="robust_z",
        config_digest="digest",
        pass_id="p1",
        risk_score=risk,
        uncertainty=0.2 if risk is not None else None,
        own_history_deviation=risk,
        herd_relative_deviation=0.1,
        own_history_window=own,
        herd_window=herd,
        derived_from_injected=injected,
    )
    base.update(kw)
    return RiskAssessment(**base)


def _record(profile):
    return feature_record(
        animal_id="a1",
        observed_at=BASE,
        pass_id="p1",
        values={"speed": 0.6, "step_asymmetry_front": 0.3},
        profile=profile,
    )


def _event(profile, config, *, risk=4.0, policy=None, **kw):
    policy = policy or ThresholdPolicy.from_config(config)
    return EventBuilder(config, policy).build(
        _assessment(risk=risk, **kw), _record(profile), window=WINDOW
    )


# -- 8.1 canonical versioned event schema -----------------------------------


def test_an_event_omitting_the_observation_window_is_refused() -> None:
    with pytest.raises(MissingFieldError) as excinfo:
        HealthEvent(
            event_id="e1",
            animal_id="a1",
            event_timestamp=BASE,
            phenotype={"speed": 0.6},
            confidence=0.8,
            contributing_sensors=("rgb-camera",),
            model_identities={"detector": "yolo11m@8.4"},
            site_key="site-a",
            level=EventLevel.ALERT,
            threshold_policy=PolicyIdentity("p0-default", "1"),
            config_digest="digest",
        )
    assert excinfo.value.field_name == "observation_window"
    assert "observation_window" in str(excinfo.value)


@pytest.mark.parametrize(
    "missing", ["animal_id", "event_timestamp", "phenotype", "site_key", "config_digest"]
)
def test_every_required_field_is_required(missing: str) -> None:
    kwargs = dict(
        event_id="e1",
        animal_id="a1",
        event_timestamp=BASE,
        observation_window=WINDOW,
        phenotype={"speed": 0.6},
        confidence=0.8,
        contributing_sensors=("rgb-camera",),
        model_identities={"detector": "yolo11m@8.4"},
        site_key="site-a",
        level=EventLevel.ALERT,
        threshold_policy=PolicyIdentity("p0-default", "1"),
        config_digest="digest",
    )
    kwargs.pop(missing)
    with pytest.raises(MissingFieldError) as excinfo:
        HealthEvent(**kwargs)
    assert excinfo.value.field_name == missing


def test_the_event_carries_the_full_minimum_contract(profile, config) -> None:
    event = _event(profile, config)
    assert event.animal_id and event.event_timestamp and event.site_key
    assert event.observation_window.seconds == 6.0
    assert event.phenotype
    assert event.contributing_sensors
    assert event.model_identities
    assert event.confidence is not None


# -- 8.2 the schema version travels with the event --------------------------


def test_the_schema_version_is_readable_from_the_event_itself(profile, config) -> None:
    encoded = _event(profile, config).to_dict()
    assert encoded["schema_name"] == "health_event"
    assert encoded["schema_version"] == HealthEvent.SCHEMA_VERSION


def test_the_schema_version_survives_export(profile, config, tmp_path) -> None:
    adapter = FileExportAdapter(tmp_path / "out")
    result = adapter.deliver(_event(profile, config))
    written = json.loads(Path(result.location).read_text(encoding="utf-8"))
    assert written["schema_version"] == HealthEvent.SCHEMA_VERSION
    assert written["schema_name"] == "health_event"


# -- 8.3 declared threshold policy ------------------------------------------


def test_an_alert_records_the_policy_that_raised_it(profile, config) -> None:
    event = _event(profile, config, risk=5.0)
    assert event.level is EventLevel.ALERT
    assert event.threshold_policy.name == config.events.threshold_policy_name
    assert event.threshold_policy.version == config.events.threshold_policy_version


def test_levels_follow_the_policy_thresholds(profile, config) -> None:
    assert _event(profile, config, risk=5.0).level is EventLevel.ALERT
    assert _event(profile, config, risk=2.5).level is EventLevel.WATCH
    assert _event(profile, config, risk=0.5).level is EventLevel.OBSERVATION


def test_an_assessment_without_a_score_cannot_cross_a_threshold(profile, config) -> None:
    policy = ThresholdPolicy.from_config(config)
    assessment = _assessment(risk=None, state=AssessmentState.INSUFFICIENT_HISTORY)
    assert policy.level_for(assessment) is EventLevel.OBSERVATION

    event = EventBuilder(config, policy).build(assessment, _record(profile), window=WINDOW)
    assert event.level is EventLevel.OBSERVATION
    assert event.risk_score is None
    assert event.assessment_state == "insufficient_history"


# -- 8.4 policy change is visible in the event stream -----------------------


def test_reprocessing_under_a_changed_policy_yields_distinguishable_alerts(profile, config) -> None:
    strict = ThresholdPolicy(
        name="p0-default", version="1", alert_threshold=3.0, watch_threshold=2.0
    )
    relaxed = ThresholdPolicy(
        name="p0-default", version="2", alert_threshold=6.0, watch_threshold=4.0
    )

    before = _event(profile, config, risk=4.0, policy=strict)
    after = _event(profile, config, risk=4.0, policy=relaxed)

    assert before.level is EventLevel.ALERT
    assert after.level is EventLevel.WATCH
    assert str(before.threshold_policy) != str(after.threshold_policy)
    # Distinguishable by the recorded policy identity alone.
    assert {str(before.threshold_policy), str(after.threshold_policy)} == {
        "p0-default@1",
        "p0-default@2",
    }


def test_a_policy_change_changes_the_idempotency_key(profile, config) -> None:
    strict = ThresholdPolicy(name="p", version="1", alert_threshold=3.0, watch_threshold=2.0)
    other = ThresholdPolicy(name="p", version="2", alert_threshold=3.0, watch_threshold=2.0)
    assert (
        _event(profile, config, policy=strict).idempotency_key
        != _event(profile, config, policy=other).idempotency_key
    )


# -- 8.5 bounded evidence retention -----------------------------------------


# A source long enough that the configured clip margins genuinely bound the
# retained clip, rather than swallowing the whole file.
SOURCE_FRAMES = 120
SOURCE_FPS = 10.0
WINDOW_FIRST = 60
WINDOW_LAST = 65


@pytest.fixture
def clip_source(tmp_path: Path) -> Path:
    return write_video(tmp_path / "media" / "lane.avi", frames=SOURCE_FRAMES, fps=SOURCE_FPS)


def _retainer(tmp_path, config, masker=None):
    return ClipRetainer(
        tmp_path / "runs",
        config,
        masker=masker if masker is not None else RegionMasker([(0.0, 0.0, 0.3, 0.3)]),
    )


def test_an_alert_retains_a_bounded_masked_clip(tmp_path, config, clip_source) -> None:
    retainer = _retainer(tmp_path, config)
    clip = retainer.retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    assert clip is not None and clip.retained
    assert Path(clip.path).exists()
    assert clip.masking_applied
    # Bounded: the window plus the configured margins, not the whole source.
    span = WINDOW_LAST - WINDOW_FIRST + 1
    expected = (
        span
        + int(config.events.clip_seconds_before * SOURCE_FPS)
        + int(config.events.clip_seconds_after * SOURCE_FPS)
    )
    assert clip.frame_count <= expected
    assert clip.frame_count < SOURCE_FRAMES, "the clip must be bounded, not the whole source"


def test_a_routine_observation_retains_no_clip(tmp_path, config, clip_source) -> None:
    retainer = _retainer(tmp_path, config)
    for level in (EventLevel.OBSERVATION, EventLevel.WATCH):
        assert (
            retainer.retain(
                level=level,
                event_id="e1",
                media_path=str(clip_source),
                media_kind="video",
                first_frame_index=15,
                last_frame_index=20,
                frame_rate=10.0,
            )
            is None
        )
    assert not (tmp_path / "runs" / "clips").exists()
    assert retainer.clips_written == 0


def test_the_retained_clip_is_reachable_from_the_event(
    tmp_path, config, profile, clip_source
) -> None:
    clip = _retainer(tmp_path, config).retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    policy = ThresholdPolicy.from_config(config)
    event = EventBuilder(config, policy).build(
        _assessment(risk=5.0), _record(profile), window=WINDOW, evidence_clip=clip
    )
    assert event.evidence_clip.retained
    assert Path(event.evidence_clip.path).exists()
    assert HealthEvent.from_dict(event.to_dict()).evidence_clip.path == clip.path


# -- 8.6 masking is applied before the clip is written ----------------------


def test_the_stored_clip_is_masked(tmp_path, config, clip_source) -> None:
    import cv2

    retainer = _retainer(tmp_path, config, masker=RegionMasker([(0.0, 0.0, 1.0, 1.0)]))
    clip = retainer.retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    stored = cv2.VideoCapture(clip.path)
    ok, masked_frame = stored.read()
    stored.release()
    assert ok

    original = cv2.VideoCapture(str(clip_source))
    original.set(
        cv2.CAP_PROP_POS_FRAMES,
        WINDOW_FIRST - int(config.events.clip_seconds_before * SOURCE_FPS),
    )
    ok, raw_frame = original.read()
    original.release()
    assert ok

    # The blurred frame is measurably smoother than the source it came from.
    assert (
        cv2.Laplacian(masked_frame, cv2.CV_64F).var() < cv2.Laplacian(raw_frame, cv2.CV_64F).var()
    )


def test_no_unmasked_copy_reaches_storage(tmp_path, config, clip_source) -> None:
    """Only the masked clip exists under the output root — nothing beside it."""
    retainer = _retainer(tmp_path, config)
    clip = retainer.retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    written = sorted(p for p in (tmp_path / "runs").rglob("*") if p.is_file())
    assert written == [Path(clip.path)]


def test_a_masker_is_required_before_anything_is_written(tmp_path, config, clip_source) -> None:
    retainer = ClipRetainer(tmp_path / "runs", config, masker=None)
    clip = retainer.retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    assert not clip.retained
    assert "without human masking" in clip.failure_reason
    assert not (tmp_path / "runs" / "clips").exists()


# -- 8.7 masking failure blocks retention -----------------------------------


class _BrokenMasker:
    @property
    def mode(self) -> str:
        return "broken"

    def mask(self, image: np.ndarray) -> np.ndarray:
        raise MaskingError("the human detector was unavailable")


def test_a_clip_that_cannot_be_masked_is_not_retained(tmp_path, config, clip_source) -> None:
    retainer = _retainer(tmp_path, config, masker=_BrokenMasker())
    clip = retainer.retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    assert not clip.retained
    assert clip.path == ""
    assert "masking failed" in clip.failure_reason
    assert not (tmp_path / "runs" / "clips").exists()
    assert retainer.retention_failures == 1


def test_the_event_records_that_evidence_retention_failed(
    tmp_path, config, profile, clip_source
) -> None:
    clip = _retainer(tmp_path, config, masker=_BrokenMasker()).retain(
        level=EventLevel.ALERT,
        event_id="e1",
        media_path=str(clip_source),
        media_kind="video",
        first_frame_index=WINDOW_FIRST,
        last_frame_index=WINDOW_LAST,
        frame_rate=SOURCE_FPS,
    )
    event = EventBuilder(config, ThresholdPolicy.from_config(config)).build(
        _assessment(risk=5.0), _record(profile), window=WINDOW, evidence_clip=clip
    )
    assert event.is_alert
    assert event.evidence_clip.retained is False
    assert "masking failed" in event.evidence_clip.failure_reason
    assert HealthEvent.from_dict(event.to_dict()).evidence_clip.failure_reason


# -- 8.8 idempotent delivery ------------------------------------------------


def test_an_event_carries_a_stable_idempotency_key(profile, config) -> None:
    first = _event(profile, config)
    second = _event(profile, config)
    assert first.idempotency_key == second.idempotency_key
    assert len(first.idempotency_key) == 32


def test_a_different_event_gets_a_different_key(profile, config) -> None:
    assert (
        _event(profile, config).idempotency_key
        != _event(profile, config, animal_id="a2").idempotency_key
    )


def test_a_redelivered_event_is_detectable_as_a_duplicate(profile, config, tmp_path) -> None:
    adapter = FileExportAdapter(tmp_path / "out")
    event = _event(profile, config)

    first = adapter.deliver(event)
    assert first.delivered and not first.duplicate

    second = adapter.deliver(event)
    assert second.delivered and second.duplicate
    assert second.key == first.key

    # One event downstream, not two.
    assert len(list((tmp_path / "out").glob("*.json"))) == 1
    assert adapter.delivered_keys() == {event.idempotency_key}


def test_the_adapter_can_be_asked_whether_it_has_seen_a_key(profile, config, tmp_path) -> None:
    adapter = FileExportAdapter(tmp_path / "out")
    event = _event(profile, config)
    assert not adapter.seen(event.idempotency_key)
    adapter.deliver(event)
    assert adapter.seen(event.idempotency_key)


# -- 8.9 export failure retains and retries ---------------------------------


class _FlakyAdapter:
    def __init__(self, *, fail_times: int) -> None:
        self.fail_times = fail_times
        self.attempts = 0
        self.delivered: list[str] = []

    @property
    def name(self) -> str:
        return "flaky"

    def seen(self, key: str) -> bool:
        return key in self.delivered

    def deliver(self, event):
        from lhv.events.export import DeliveryResult

        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConsumerUnavailable("consumer is down")
        self.delivered.append(event.idempotency_key)
        return DeliveryResult(key=event.idempotency_key, delivered=True)


def test_an_unavailable_consumer_causes_retention_and_retry(profile, config, tmp_path) -> None:
    adapter = _FlakyAdapter(fail_times=1)
    exporter = EventExporter(adapter, retention_dir=tmp_path / "pending", max_attempts=3)
    report = exporter.export([_event(profile, config)])

    assert adapter.attempts == 2
    assert report.retries == 1
    assert report.delivered == 1
    assert report.undelivered == 0
    assert exporter.pending() == []


def test_an_event_that_never_delivers_is_retained_and_counted(profile, config, tmp_path) -> None:
    adapter = _FlakyAdapter(fail_times=99)
    exporter = EventExporter(adapter, retention_dir=tmp_path / "pending", max_attempts=3)
    event = _event(profile, config)
    report = exporter.export([event])

    assert report.undelivered == 1
    assert report.delivered == 0
    assert "1 undelivered" in report.describe()
    # Retained, not dropped.
    assert exporter.pending() == [event.idempotency_key]
    assert (tmp_path / "pending" / f"{event.idempotency_key}.json").exists()


def test_a_later_run_delivers_what_an_earlier_one_retained(profile, config, tmp_path) -> None:
    event = _event(profile, config)

    down = _FlakyAdapter(fail_times=99)
    EventExporter(down, retention_dir=tmp_path / "pending", max_attempts=2).export([event])

    up = FileExportAdapter(tmp_path / "out")
    recovered = EventExporter(up, retention_dir=tmp_path / "pending", max_attempts=2)
    report = recovered.retry_pending({event.idempotency_key: event})

    assert report.delivered == 1
    assert recovered.pending() == []
    assert up.seen(event.idempotency_key)


def test_the_file_adapter_reports_being_unavailable(profile, config, tmp_path) -> None:
    adapter = FileExportAdapter(tmp_path / "out")
    adapter.available = False
    with pytest.raises(ConsumerUnavailable):
        adapter.deliver(_event(profile, config))


# -- 8.10 stub-derived marking ----------------------------------------------


def test_every_p0_event_is_marked_non_clinical_and_stub_derived(profile, config) -> None:
    for risk in (0.5, 2.5, 5.0):
        event = _event(profile, config, risk=risk)
        assert event.stub_derived
        assert event.non_clinical
        assert any("not clinical evidence" in note for note in event.notes)


def test_an_event_may_not_be_stub_derived_and_clinical_at_once() -> None:
    with pytest.raises(ValueError, match="may not be presented as a clinical finding"):
        HealthEvent(
            event_id="e1",
            animal_id="a1",
            event_timestamp=BASE,
            observation_window=WINDOW,
            phenotype={},
            confidence=0.5,
            contributing_sensors=("rgb-camera",),
            model_identities={},
            site_key="site-a",
            level=EventLevel.ALERT,
            threshold_policy=PolicyIdentity("p", "1"),
            config_digest="d",
            stub_derived=True,
            non_clinical=False,
        )


def test_an_injected_event_says_so(profile, config) -> None:
    event = _event(profile, config, injected=True)
    assert event.derived_from_injected
    assert any("synthetic deviation" in note for note in event.notes)


def test_the_markers_survive_export(profile, config, tmp_path) -> None:
    adapter = FileExportAdapter(tmp_path / "out")
    result = adapter.deliver(_event(profile, config, injected=True))
    written = json.loads(Path(result.location).read_text(encoding="utf-8"))
    assert written["stub_derived"] is True
    assert written["non_clinical"] is True
    assert written["derived_from_injected"] is True
