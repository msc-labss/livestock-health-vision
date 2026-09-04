"""Baseline: the time series, own-history and herd baselines, cold start, risk, injection."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timedelta

import pytest

from lhv.baseline import (
    AssessmentState,
    BaselineEngine,
    InjectionShape,
    InjectionSpec,
    RiskAssessment,
    TimeSeriesStore,
    inject,
)
from lhv.profiles import load_profile

from .conftest import feature_record

BASE = datetime(2024, 3, 1, 6, 30, tzinfo=UTC)


@pytest.fixture
def profile():
    return load_profile("cattle")


@pytest.fixture
def store(tmp_path, profile):
    return TimeSeriesStore(tmp_path / "series", profile.feature_set)


@pytest.fixture
def engine(store, profile, config):
    return BaselineEngine(store, profile, config)


def _record(profile, animal_id, day, **values):
    defaults = {"speed": 0.75, "lateral_sway": 0.02, "step_asymmetry_front": 0.05}
    defaults.update(values)
    return feature_record(
        animal_id=animal_id,
        observed_at=BASE + timedelta(days=day, seconds=animal_id.__hash__() % 60),
        pass_id=f"{animal_id}-d{day}",
        values=defaults,
        profile=profile,
    )


def _history(profile, animal_id="a1", days=10, **values):
    # A little variation, so a robust spread is defined.
    return [
        _record(
            profile,
            animal_id,
            day,
            speed=0.75 + 0.01 * (day % 3),
            lateral_sway=0.02 + 0.001 * (day % 4),
            step_asymmetry_front=0.05 + 0.002 * (day % 5),
            **values,
        )
        for day in range(days)
    ]


# -- 7.1 time series append -------------------------------------------------


def test_a_valid_resolved_pass_is_appended_at_its_pass_timestamp(store, profile) -> None:
    record = _record(profile, "a1", 0)
    result = store.append([record])
    assert result.appended == 1

    observations = store.read(animal_id="a1")
    assert len(observations) == 1
    assert observations[0].observed_at == record.observed_at
    assert observations[0].value("speed") == pytest.approx(0.75)


def test_the_store_is_partitioned_by_site_and_day(store, profile) -> None:
    store.append([_record(profile, "a1", 0), _record(profile, "a1", 1)])
    partitions = sorted(p.name for p in store.area("observations").rglob("day_key=*"))
    assert partitions == ["day_key=2024-03-01", "day_key=2024-03-02"]
    assert (store.area("observations") / "site_key=site-a").is_dir()


def test_observations_read_back_in_time_order(store, profile) -> None:
    store.append(list(reversed(_history(profile, days=5))))
    observations = store.read(animal_id="a1")
    assert [o.observed_at for o in observations] == sorted(o.observed_at for o in observations)


def test_the_store_can_be_queried_by_window(store, profile) -> None:
    store.append(_history(profile, days=10))
    window = store.read(
        animal_id="a1", start=BASE + timedelta(days=3), end=BASE + timedelta(days=6)
    )
    assert 0 < len(window) < 10


# -- 7.2 unresolved identity does not enter any series ----------------------


def test_a_valid_pass_with_unresolved_identity_modifies_no_time_series(store, profile) -> None:
    resolved = _record(profile, "a1", 0)
    unattributed = dataclasses.replace(_record(profile, "a1", 1), animal_id="")

    result = store.append([resolved, unattributed])
    assert result.appended == 1
    assert result.unattributed == 1

    series = store.read(animal_id="a1")
    assert [o.pass_id for o in series] == [resolved.pass_id]


def test_an_unattributed_pass_lands_in_the_unattributed_store(store, profile) -> None:
    unattributed = dataclasses.replace(_record(profile, "a1", 0), animal_id="")
    store.append([unattributed])

    assert store.read(area="observations") == []
    held = store.read(area="unattributed")
    assert [o.pass_id for o in held] == [unattributed.pass_id]


def test_an_invalid_pass_is_retained_for_audit_and_not_in_the_series(store, profile) -> None:
    invalid = dataclasses.replace(_record(profile, "a1", 0), valid=False)
    result = store.append([invalid])
    assert result.audited == 1
    assert store.read(area="observations") == []
    assert len(store.read(area="audit")) == 1


# -- 7.3 own-history baseline -----------------------------------------------


def test_the_baseline_excludes_the_observation_under_test(store, engine, profile) -> None:
    store.append(_history(profile, days=10))
    observations = store.read(animal_id="a1")
    scored = observations[-1]

    baseline = engine.own_history(scored)
    assert baseline.window.observation_count == len(observations) - 1
    assert scored.pass_id in baseline.window.excluded_pass_ids


def test_an_extreme_observation_does_not_flatten_its_own_baseline(store, engine, profile) -> None:
    """Were the scored point included, a large deviation would partly hide itself."""
    store.append(_history(profile, days=10))
    store.append([_record(profile, "a1", 10, speed=0.20)])

    observations = store.read(animal_id="a1")
    scored = observations[-1]
    baseline = engine.own_history(scored)
    speed = baseline.get("speed")
    assert speed.count == len(observations) - 1
    assert speed.centre == pytest.approx(0.75, abs=0.03)


def test_the_lookback_window_bounds_the_baseline(store, profile, config) -> None:
    short = dataclasses.replace(
        config, baseline=dataclasses.replace(config.baseline, lookback_days=3)
    )
    store.append(_history(profile, days=10))
    engine = BaselineEngine(store, load_profile("cattle"), short)
    scored = store.read(animal_id="a1")[-1]

    baseline = engine.own_history(scored)
    assert baseline.window.lookback_days == 3
    assert baseline.window.observation_count < 9


# -- 7.4 herd baseline ------------------------------------------------------


def _herd(profile, *, animals=("a1", "a2", "a3", "a4"), days=10, shift_from=None, shift=0.0):
    records = []
    for animal in animals:
        for day in range(days):
            offset = shift if shift_from is not None and day >= shift_from else 0.0
            records.append(
                _record(
                    profile,
                    animal,
                    day,
                    speed=0.75 + 0.01 * ((day + len(animal)) % 3) + offset,
                    lateral_sway=0.02 + 0.001 * (day % 4) + offset * 0.1,
                    step_asymmetry_front=0.05 + 0.002 * (day % 5),
                )
            )
    return records


def test_a_herd_wide_shift_moves_own_history_but_not_herd_relative_deviation(
    store, engine, profile
) -> None:
    """Everybody slows down together: that is a herd event, not four lame animals.

    The shift is in place across the whole herd window and across only the tail
    of the own-history lookback, which is exactly the situation the two
    baselines exist to tell apart.
    """
    store.append(_herd(profile, days=12, shift_from=8, shift=-0.30))
    scored = [o for o in store.read(animal_id="a1") if o.observed_at >= BASE + timedelta(days=11)][
        0
    ]

    assessment = engine.assess(scored)
    assert assessment.scored
    assert assessment.own_history_deviation is not None
    assert assessment.herd_relative_deviation is not None

    # Own history sees a large move; the contemporaneous herd sees almost none.
    assert abs(assessment.own_history_deviation) > 3.0
    assert abs(assessment.herd_relative_deviation) < 1.0
    assert abs(assessment.herd_relative_deviation) < abs(assessment.own_history_deviation)


def test_both_deviations_are_reported_side_by_side(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    scored = store.read(animal_id="a1")[-1]
    encoded = engine.assess(scored).to_dict()
    assert "own_history_deviation" in encoded
    assert "herd_relative_deviation" in encoded


def test_the_herd_window_records_how_many_animals_it_covered(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    scored = store.read(animal_id="a1")[-1]
    herd = engine.herd(scored)
    assert herd.window.animal_count >= 3
    assert herd.window.site_key == "site-a"


# -- 7.5 cold start ---------------------------------------------------------


def test_an_animal_below_the_minimum_yields_insufficient_history(store, engine, profile) -> None:
    store.append(_history(profile, animal_id="new", days=3))
    scored = store.read(animal_id="new")[-1]

    assessment = engine.assess(scored)
    assert assessment.state is AssessmentState.INSUFFICIENT_HISTORY
    assert assessment.risk_score is None
    assert (
        assessment.shortfall == assessment.observations_required - assessment.observations_available
    )
    assert "required" in assessment.note


def test_insufficient_history_is_not_a_low_score(store, engine, profile) -> None:
    store.append(_history(profile, animal_id="new", days=2))
    assessment = engine.assess(store.read(animal_id="new")[-1])
    assert not assessment.scored
    assert assessment.risk_score is None


def test_a_scored_assessment_may_not_omit_its_score() -> None:
    with pytest.raises(ValueError, match="must carry a risk score"):
        RiskAssessment(
            assessment_id="x",
            animal_id="a1",
            observed_at=BASE,
            site_key="site-a",
            day_key="2024-03-01",
            state=AssessmentState.SCORED,
            scale="robust_z",
            config_digest="d",
        )


def test_insufficient_history_may_not_carry_a_score() -> None:
    with pytest.raises(ValueError, match="must not carry a numeric risk score"):
        RiskAssessment(
            assessment_id="x",
            animal_id="a1",
            observed_at=BASE,
            site_key="site-a",
            day_key="2024-03-01",
            state=AssessmentState.INSUFFICIENT_HISTORY,
            scale="robust_z",
            config_digest="d",
            risk_score=0.1,
        )


# -- 7.6 risk score carries uncertainty and its inputs ----------------------


def test_the_score_carries_an_uncertainty_measure(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    assessment = engine.assess(store.read(animal_id="a1")[-1])
    assert assessment.uncertainty is not None
    assert assessment.uncertainty >= 0.0


def test_the_score_names_the_baselines_and_windows_it_used(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    assessment = engine.assess(store.read(animal_id="a1")[-1])

    assert assessment.own_history_window is not None
    assert assessment.herd_window is not None
    assert assessment.own_history_window.lookback_days == 21
    assert assessment.herd_window.lookback_days == 3
    assert "own_history:a1" in assessment.own_history_window.reference
    assert assessment.scale == "robust_z"
    assert assessment.config_digest


def test_the_score_is_recomputable_from_its_recorded_inputs(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    original = engine.assess(store.read(animal_id="a1")[-1])

    recomputed = engine.recompute(original)
    assert recomputed.risk_score == pytest.approx(original.risk_score)
    assert recomputed.uncertainty == pytest.approx(original.uncertainty)
    assert recomputed.own_history_deviation == pytest.approx(original.own_history_deviation)


def test_the_score_survives_its_own_encoding(store, engine, profile) -> None:
    store.append(_herd(profile, days=12))
    assessment = engine.assess(store.read(animal_id="a1")[-1])
    restored = RiskAssessment.from_dict(assessment.to_dict())
    assert restored.risk_score == pytest.approx(assessment.risk_score)
    assert restored.own_history_window.reference == assessment.own_history_window.reference


# -- 7.7 injection ----------------------------------------------------------


def test_an_injected_deviation_of_known_magnitude_moves_the_risk_score(
    store, engine, profile
) -> None:
    store.append(_history(profile, days=10))
    clean = _record(profile, "a1", 10)

    store.append([clean])
    baseline_score = engine.assess(store.read(animal_id="a1")[-1]).risk_score

    other = TimeSeriesStore(store.root.parent / "injected", profile.feature_set)
    other.append(_history(profile, days=10))
    injected = inject(
        [clean],
        [
            InjectionSpec(
                feature="step_asymmetry_front",
                magnitude=0.40,
                shape=InjectionShape.STEP,
                starts_at=BASE + timedelta(days=9, hours=12),
                animal_id="a1",
            )
        ],
    )
    other.append(injected)
    injected_engine = BaselineEngine(other, profile, engine.config)
    injected_score = injected_engine.assess(other.read(animal_id="a1")[-1]).risk_score

    assert injected_score > baseline_score
    assert injected_score - baseline_score > 1.0


@pytest.mark.parametrize(
    "shape",
    [InjectionShape.STEP, InjectionShape.RAMP, InjectionShape.SPIKE, InjectionShape.TRANSIENT],
)
def test_every_declared_shape_applies_within_its_window(profile, shape) -> None:
    spec = InjectionSpec(
        feature="speed",
        magnitude=1.0,
        shape=shape,
        starts_at=BASE,
        ends_at=BASE + timedelta(days=4),
    )
    assert spec.weight(BASE - timedelta(days=1)) == 0.0
    mid = spec.weight(BASE + timedelta(days=2))
    assert 0.0 < mid <= 1.0


def test_a_ramp_grows_across_its_window(profile) -> None:
    spec = InjectionSpec(
        feature="speed",
        magnitude=1.0,
        shape=InjectionShape.RAMP,
        starts_at=BASE,
        ends_at=BASE + timedelta(days=4),
    )
    assert spec.weight(BASE + timedelta(days=1)) < spec.weight(BASE + timedelta(days=3))


def test_a_transient_returns_to_zero(profile) -> None:
    spec = InjectionSpec(
        feature="speed",
        magnitude=1.0,
        shape=InjectionShape.TRANSIENT,
        starts_at=BASE,
        ends_at=BASE + timedelta(days=4),
    )
    assert spec.weight(BASE + timedelta(days=2)) == pytest.approx(1.0)
    assert spec.weight(BASE + timedelta(days=5)) == 0.0


def test_the_injected_magnitude_is_exactly_what_was_declared(profile) -> None:
    clean = _record(profile, "a1", 0, speed=0.75)
    injected = inject(
        [clean],
        [
            InjectionSpec(
                feature="speed",
                magnitude=-0.30,
                shape=InjectionShape.STEP,
                starts_at=BASE - timedelta(days=1),
            )
        ],
    )[0]
    assert injected.value("speed") == pytest.approx(0.45)


# -- 7.8 injection marking travels ------------------------------------------


def test_an_injected_record_is_marked(profile) -> None:
    clean = _record(profile, "a1", 0)
    assert not clean.injected
    injected = inject(
        [clean],
        [
            InjectionSpec(
                feature="speed",
                magnitude=0.5,
                shape=InjectionShape.STEP,
                starts_at=BASE - timedelta(days=1),
            )
        ],
    )[0]
    assert injected.injected
    assert "injected offset" in next(f for f in injected.features if f.name == "speed").note


def test_a_record_outside_the_injection_window_is_not_marked(profile) -> None:
    clean = _record(profile, "a1", 0)
    untouched = inject(
        [clean],
        [
            InjectionSpec(
                feature="speed",
                magnitude=0.5,
                shape=InjectionShape.STEP,
                starts_at=BASE + timedelta(days=30),
            )
        ],
    )[0]
    assert not untouched.injected
    assert untouched.value("speed") == pytest.approx(clean.value("speed"))


def test_the_marker_survives_the_store_and_reaches_the_assessment(store, engine, profile) -> None:
    store.append(_history(profile, days=10))
    injected = inject(
        [_record(profile, "a1", 10)],
        [
            InjectionSpec(
                feature="step_asymmetry_front",
                magnitude=0.4,
                shape=InjectionShape.STEP,
                starts_at=BASE,
                animal_id="a1",
            )
        ],
    )
    store.append(injected)

    observation = store.read(animal_id="a1")[-1]
    assert observation.injected, "the marker must survive the columnar round trip"

    assessment = engine.assess(observation)
    assert assessment.derived_from_injected


def test_an_assessment_from_clean_data_is_not_marked(store, engine, profile) -> None:
    store.append(_history(profile, days=12))
    assessment = engine.assess(store.read(animal_id="a1")[-1])
    assert not assessment.derived_from_injected


def test_every_p0_assessment_declares_itself_stub_derived(store, engine, profile) -> None:
    store.append(_history(profile, days=12))
    assessment = engine.assess(store.read(animal_id="a1")[-1])
    assert assessment.stub_derived


# -- a source that records no capture time ----------------------------------


def test_a_pass_without_a_timestamp_is_retained_rather_than_crashing(store, profile) -> None:
    """Pre-cropped stills carry no clock; the store must still accept them.

    They cannot enter a series — a series is keyed by observation time — but
    refusing to write them at all would lose the pass, and inventing a time
    would put it in the series under a fabricated key.
    """
    import dataclasses

    undated = dataclasses.replace(_record(profile, "a1", 0), observed_at=None)
    result = store.append([undated])

    assert result.appended == 0
    assert result.skipped_without_timestamp == 1
    assert store.read(area="observations") == []

    held = store.read(area="unattributed")
    assert [o.pass_id for o in held] == [undated.pass_id]
    assert held[0].observed_at is None


def test_undated_passes_do_not_disturb_dated_ones(store, profile) -> None:
    import dataclasses

    dated = _record(profile, "a1", 0)
    undated = dataclasses.replace(_record(profile, "a1", 1), observed_at=None)
    result = store.append([dated, undated])

    assert result.appended == 1
    assert result.skipped_without_timestamp == 1
    assert [o.pass_id for o in store.read(animal_id="a1")] == [dated.pass_id]
