"""Checks that run against the real public datasets, when they are on disk.

Marked ``dataset`` and skipped otherwise, so continuous integration stays
runnable without 40 GB of livestock footage. These are the checks that turn the
registrations' declared figures into verified ones.

They run against ``cattle-topdown``, not the default profile. CattleEyeView is
overhead footage and the default profile is lateral; pose refuses that pairing
by design, because a skeleton's view decides what its keypoints mean. Naming the
matching profile here is the point of the seam, and it keeps P0's findings —
above all that a top-down camera cannot see a cow's legs — reproducible on
demand rather than only recorded in a document.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from lhv.config import ResolvedConfig
from lhv.datasets import load_registration
from lhv.ingest import Ingestor

pytestmark = pytest.mark.dataset

DATA = Path(__file__).resolve().parent.parent / "data"
CEV = DATA / "cattleeyeview"


def _require(path: Path, what: str) -> Path:
    if not path.exists():
        pytest.skip(f"{what} not present at {path}")
    return path


@pytest.fixture
def cattleeyeview():
    _require(CEV / "videos", "CattleEyeView videos")
    return load_registration("cattleeyeview")


@pytest.fixture
def config(cattleeyeview):
    return ResolvedConfig(
        species_profile="cattle",
        species_profile_version="0",
        dataset_name=cattleeyeview.name,
        dataset_version=cattleeyeview.version,
    )


# -- content counts, checked against the download rather than the paper -------


def test_the_sequence_count_matches_the_paper(cattleeyeview) -> None:
    report = cattleeyeview.verify(CEV)
    sequences = next(c for c in report.counts if c.name == "sequences")
    assert sequences.observed == 14
    assert sequences.matches


def test_the_extracted_frame_count_matches_the_paper(cattleeyeview) -> None:
    """Counted inside images.tar.gz without unpacking its 12 GB."""
    if not (CEV / "images.tar.gz").exists():
        pytest.skip("images.tar.gz not present")
    report = cattleeyeview.verify(CEV)
    frames = next(c for c in report.counts if c.name == "frames")
    assert frames.declared == 30703
    assert frames.observed == 30703
    assert frames.matches


def test_the_videos_hold_three_frames_the_extracted_set_does_not(cattleeyeview) -> None:
    """Both figures are right and measure different things.

    The paper's 30,703 is the extracted set. The videos decode to 30,706,
    because the final frame of sequences 03, 05 and 06 was never extracted.
    """
    report = cattleeyeview.verify(CEV)
    in_videos = next(c for c in report.counts if c.name == "frames_in_videos")
    assert in_videos.observed == 30706
    assert in_videos.matches, "the measured figure is recorded, so this is a regression check"

    if not (CEV / "images.tar.gz").exists():
        pytest.skip("images.tar.gz not present")
    extracted = next(c for c in report.counts if c.name == "frames")
    assert in_videos.observed - extracted.observed == 3


def test_the_release_image_paths_resolve_to_video_frames(cattleeyeview) -> None:
    """images/<sequence>.mp4/<frame>.jpg, one-based against a zero-based video."""
    layout = cattleeyeview.layout("frames")
    captures = layout.match("images/03.mp4/00871.jpg")
    assert captures == {"sequence": "03", "frame": "00871"}
    assert int(captures["frame"]) - 1 == 870


# -- the recovered recording times -------------------------------------------


def test_every_sequence_has_a_recording_time(cattleeyeview) -> None:
    sources = cattleeyeview.sources("footage", CEV)
    assert len(sources) == 14
    assert all(s.start_timestamp is not None for s in sources)


def test_the_recording_window_matches_what_the_paper_states(cattleeyeview) -> None:
    """The paper says 2021-11-09 to 2022-03-04; the burned-in overlays agree."""
    sources = sorted(cattleeyeview.sources("footage", CEV), key=lambda s: s.source_id)
    assert sources[0].start_timestamp.date() == date(2021, 11, 9)
    assert sources[-1].start_timestamp.date() == date(2022, 3, 4)


def test_recording_times_increase_with_the_sequence_number(cattleeyeview) -> None:
    sources = sorted(cattleeyeview.sources("footage", CEV), key=lambda s: s.source_id)
    stamps = [s.start_timestamp for s in sources]
    assert stamps == sorted(stamps)


def test_the_sequences_span_thirteen_distinct_days(cattleeyeview) -> None:
    """Two sequences share 2021-12-19, so a day split is not a sequence split."""
    sources = cattleeyeview.sources("footage", CEV)
    days = {s.day_key for s in sources}
    assert len(days) == 13
    assert len(sources) == 14


# -- ingest over real media ---------------------------------------------------


def test_ingest_emits_every_frame_of_a_real_sequence(cattleeyeview, config) -> None:
    source = next(s for s in cattleeyeview.sources("footage", CEV) if s.source_id.endswith("/01"))
    frames = list(Ingestor(source, config).iter_frames(decode=False))
    assert len(frames) == 1226
    assert [f.index for f in frames] == list(range(1226))


def test_real_frames_carry_reliable_capture_times(cattleeyeview, config) -> None:
    source = next(s for s in cattleeyeview.sources("footage", CEV) if s.source_id.endswith("/01"))
    frames = list(Ingestor(source, config).iter_frames(decode=False))

    assert all(f.provenance.timestamp_reliable for f in frames)
    assert frames[0].provenance.capture_timestamp == source.start_timestamp
    # 8 fps: an eighth of a second per frame.
    step = (
        frames[1].provenance.capture_timestamp - frames[0].provenance.capture_timestamp
    ).total_seconds()
    assert step == pytest.approx(0.125)
    assert all(f.provenance.day_key == "2021-11-09" for f in frames)


def test_real_frames_carry_the_registration_split_keys(cattleeyeview, config) -> None:
    source = cattleeyeview.sources("footage", CEV)[0]
    frame = next(iter(Ingestor(source, config).iter_frames(decode=False)))
    keys = frame.provenance.split_keys()
    assert keys["site"] == cattleeyeview.site_key
    assert keys["camera"] in cattleeyeview.camera_ids
    assert keys["day"] == "2021-11-09"


def test_ingest_over_a_real_sequence_is_deterministic(cattleeyeview, config) -> None:
    source = cattleeyeview.sources("footage", CEV)[0]
    first = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    second = [f.provenance for f in Ingestor(source, config).iter_frames(decode=False)]
    assert first == second


# -- the sampling rates that limit what can be measured -----------------------


def test_the_sequences_are_1920x1080_at_three_declared_frame_rates(cattleeyeview) -> None:
    """The paper states neither. Both bound what the phenotype layer can report."""
    import cv2

    rates, sizes = set(), set()
    for source in cattleeyeview.sources("footage", CEV):
        capture = cv2.VideoCapture(source.media_path)
        try:
            rates.add(round(capture.get(cv2.CAP_PROP_FPS)))
            sizes.add(
                (
                    int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                )
            )
        finally:
            capture.release()

    assert sizes == {(1920, 1080)}
    assert rates == {3, 5, 8}


def test_the_slow_sequences_cannot_resolve_a_stride(cattleeyeview) -> None:
    """Five of fourteen sequences run at 3 fps, below the profile's requirement."""
    import cv2

    from lhv.profiles import load_profile

    required = load_profile("cattle-topdown").feature_set.get("stride_frequency_front")
    assert required.requires_sampling_hz == 5.0

    too_slow = 0
    for source in cattleeyeview.sources("footage", CEV):
        capture = cv2.VideoCapture(source.media_path)
        try:
            if capture.get(cv2.CAP_PROP_FPS) < required.requires_sampling_hz:
                too_slow += 1
        finally:
            capture.release()
    assert too_slow == 5


# -- the default profile refuses this footage, by design ---------------------


def test_the_lateral_profile_refuses_this_top_down_footage(cattleeyeview) -> None:
    """The guard the lateral profile exists for, demonstrated on real footage.

    P0's own finding is that a top-down camera cannot see a cow's legs. The
    lateral profile is the response, and pointing it at this footage would
    produce keypoints whose names claim a geometry the footage does not have.
    Refusing is the whole point, and it is checked here against the real source
    rather than a synthetic one.
    """
    from lhv.config import ResolvedConfig
    from lhv.errors import ViewMismatchError
    from lhv.perception import AnnotationPoseBackend, PoseEstimator
    from lhv.profiles import load_profile

    lateral = load_profile("cattle")
    assert lateral.skeleton.view == "lateral"

    source = cattleeyeview.sources("footage", CEV)[0]
    assert source.view == "top-down"

    config = ResolvedConfig(
        species_profile=lateral.species,
        species_profile_version=lateral.version,
        dataset_name=cattleeyeview.name,
        dataset_version=cattleeyeview.version,
        models={},
        feature_set_version=lateral.feature_set.version,
    )
    with pytest.raises(ViewMismatchError) as raised:
        PoseEstimator(
            AnnotationPoseBackend({}),
            lateral,
            config,
            source_id=source.source_id,
            source_view=source.view,
        )
    message = str(raised.value)
    assert "top-down" in message and "lateral" in message
    assert source.source_id in message


# -- the pipeline over real CattleEyeView footage ----------------------------


@pytest.fixture(scope="module")
def cev_run(tmp_path_factory):
    """Registration through to exported events, over two real sequences."""
    import dataclasses
    from datetime import timedelta

    import cv2

    from lhv.config import ModelIdentity, ResolvedConfig
    from lhv.datasets import LabelledPoseBackend, load_coco_keypoints, load_registration
    from lhv.identity import AnchorRecord, DatasetLabelAnchorSource
    from lhv.perception import AnnotationDetector
    from lhv.pipeline import Pipeline
    from lhv.profiles import load_profile

    if not (CEV / "videos").exists() or not (CEV / "annotation/pose_COCO").exists():
        pytest.skip("CattleEyeView videos or annotations not present")

    profile = load_profile("cattle-topdown")
    registration = load_registration("cattleeyeview")
    sources = [
        s
        for s in registration.sources("footage", CEV)
        if s.source_id.rsplit("/", 1)[-1] in {"01", "05"}
    ]

    labels = load_coco_keypoints(
        CEV / "annotation/pose_COCO/coco_track_test.json",
        dataset_name=registration.name,
        skeleton=profile.skeleton,
    )

    base = ResolvedConfig(
        species_profile="x", species_profile_version="0", dataset_name="x", dataset_version="1"
    )
    config = ResolvedConfig(
        species_profile=profile.species,
        species_profile_version=profile.version,
        dataset_name=registration.name,
        dataset_version=registration.version,
        seed=7,
        models={
            "detector": ModelIdentity(name="dataset-box-label", version="1", task="detect"),
            "pose": ModelIdentity(name="dataset-keypoint-label", version="1", task="pose"),
        },
        perception=dataclasses.replace(
            base.perception, detector_backend="injected", pose_backend="injected"
        ),
        phenotype=dataclasses.replace(
            base.phenotype,
            boundary_axis="x",
            entry_boundary=0.3,
            exit_boundary=0.7,
            min_pass_frames=6,
        ),
        baseline=dataclasses.replace(base.baseline, lookback_days=120, min_observations=3),
    )

    rates, spans, extents = {}, {}, {}
    for source in sources:
        capture = cv2.VideoCapture(source.media_path)
        rates[source.source_id] = capture.get(cv2.CAP_PROP_FPS) or 8.0
        capture.release()
    for box in labels.boxes:
        key = (box.source_id, box.track_id)
        low, high = spans.get(key, (box.frame_index, box.frame_index))
        spans[key] = (min(low, box.frame_index), max(high, box.frame_index))
        x1, y1, x2, y2 = extents.get(key, (1e9, 1e9, -1e9, -1e9))
        extents[key] = (
            min(x1, box.box.x1),
            min(y1, box.box.y1),
            max(x2, box.box.x2),
            max(y2, box.box.y2),
        )

    by_source = {s.source_id: s for s in sources}
    anchors = []
    for (source_id, track), (low, high) in spans.items():
        source = by_source.get(source_id)
        if source is None:
            continue
        rate = rates[source_id]
        anchors.append(
            AnchorRecord(
                animal_id=f"instance-{track}",
                anchor_source="dataset-label:cattleeyeview",
                site_key=source.site_key,
                camera_id=source.camera_id,
                reader_id="ground-truth",
                observed_from=source.start_timestamp + timedelta(seconds=low / rate),
                observed_to=source.start_timestamp + timedelta(seconds=high / rate),
                region=tuple(
                    v / d
                    for v, d in zip(
                        extents[(source_id, track)], (1920, 1080, 1920, 1080), strict=True
                    )
                ),
            )
        )

    pipeline = Pipeline(
        config,
        profile,
        tmp_path_factory.mktemp("cev-run"),
        detector_backend=AnnotationDetector(labels.detector_boxes()),
        pose_backend=LabelledPoseBackend(labels),
        anchor_source=DatasetLabelAnchorSource(anchors, dataset_name="cattleeyeview"),
    )
    result = pipeline.run(sources, frame_width=1920, frame_height=1080)
    return pipeline, result


def test_the_pipeline_reaches_exported_events_on_real_footage(cev_run) -> None:
    pipeline, result = cev_run
    assert result.sources == 2
    assert sum(r.detections_emitted for r in result.perception) > 0
    assert sum(r.tracklets_formed for r in result.perception) > 0
    assert result.valid_passes > 0
    assert result.observations > 0
    assert result.events > 0
    assert result.exported == result.events
    assert result.undelivered == 0


def test_tracklet_identifiers_are_unique_within_every_source(cev_run) -> None:
    """4.5, over a real pass through the source rather than a synthetic one."""
    import collections

    from lhv.perception import Tracklet

    pipeline, _ = cev_run
    tracklets = pipeline.store.read("tracklets", Tracklet)
    assert tracklets

    per_source = collections.defaultdict(list)
    for tracklet in tracklets:
        per_source[tracklet.source_id].append(tracklet.tracklet_id)
    for source_id, identifiers in per_source.items():
        assert len(identifiers) == len(set(identifiers)), f"duplicates within {source_id}"
    # They happen to be unique across sources too, because the source id is in them.
    assert len({t.tracklet_id for t in tracklets}) == len(tracklets)


def test_every_assessment_reports_insufficient_history(cev_run) -> None:
    """CattleEyeView labels instances, not individuals followed across days.

    An instance appears in exactly one sequence, so no animal ever accumulates
    the history a baseline needs. The right answer is to say so, not to score.
    """
    from lhv.baseline import AssessmentState, RiskAssessment

    pipeline, result = cev_run
    assessments = pipeline.store.read("assessments", RiskAssessment)
    assert assessments
    assert all(a.state is AssessmentState.INSUFFICIENT_HISTORY for a in assessments)
    assert all(a.risk_score is None for a in assessments)
    assert result.scored == 0


def test_every_exported_event_is_marked_non_clinical(cev_run) -> None:
    from lhv.events import HealthEvent

    pipeline, _ = cev_run
    events = pipeline.store.read("events", HealthEvent)
    assert events
    assert all(e.stub_derived and e.non_clinical for e in events)


def test_the_limb_features_are_unusable_on_a_top_down_source(cev_run) -> None:
    """A camera directly overhead cannot see the legs under the animal.

    Paws are labelled visible in under 10% of instances, so every feature that
    depends on one is correctly reported unusable rather than computed from
    almost nothing.
    """
    from lhv.phenotype import FeatureRecord, QualityFlag

    pipeline, _ = cev_run
    records = pipeline.store.read("features", FeatureRecord)
    assert records

    limb = ("stride_length_front", "stride_frequency_front", "step_asymmetry_front")
    body = ("speed", "lateral_sway", "spine_lateral_curvature")

    for name in limb:
        qualities = [r.quality(name) for r in records]
        unusable = sum(1 for q in qualities if q is QualityFlag.UNUSABLE)
        assert unusable / len(qualities) > 0.9, f"{name} should be unusable on a top-down view"

    for name in body:
        qualities = [r.quality(name) for r in records]
        good = sum(1 for q in qualities if q is QualityFlag.GOOD)
        assert good > 0, f"{name} should be measurable from the body axis"


# -- MultiCamCows2024 --------------------------------------------------------

MCC = DATA / "multicamcows2024"


@pytest.fixture
def multicamcows():
    _require(MCC / "MultiCamCows2024Root", "MultiCamCows2024 imagery")
    return load_registration("multicamcows2024")


def test_the_counts_match_what_the_publishers_state(multicamcows) -> None:
    report = multicamcows.verify(MCC)
    observed = {c.name: (c.observed, c.declared) for c in report.counts}
    assert observed["images"] == (101329, 101329)
    assert observed["days"] == (7, 7)
    if (MCC / "archive.zip").exists():
        assert observed["videos_in_archive"] == (137, 137)


def test_identity_labels_and_camera_partitioning_are_usable(multicamcows) -> None:
    """2.2: both are present in the paths, and the layout resolves them."""
    sources = multicamcows.sources("tracklets", MCC)
    assert len(sources) == 1584
    assert len({s.animal_id for s in sources}) == 90
    assert {s.camera_id for s in sources} == set(multicamcows.camera_ids)
    assert len({s.day_key for s in sources}) == 7
    assert sum(len(s.media_paths) for s in sources) == 101329
    # Every source is one animal, one day, one camera.
    for source in sources[:20]:
        assert source.animal_id
        assert all(p.endswith(f"_{source.camera_id[-1]}.jpg") for p in source.media_paths)


def test_the_stills_carry_no_capture_time_and_say_so(multicamcows, config) -> None:
    """A series is keyed by observation time, so this source cannot enter one."""
    sources = multicamcows.sources("tracklets", MCC)
    assert all(s.start_timestamp is None for s in sources[:50])

    ingestor = Ingestor(sources[0], config)
    frames = list(ingestor.iter_frames(decode=False))
    assert frames
    assert all(f.provenance.capture_timestamp is None for f in frames)
    assert all(not f.provenance.timestamp_reliable for f in frames)
    # The day key still exists, taken from the path rather than invented.
    assert all(f.provenance.day_key == sources[0].day_key for f in frames)
