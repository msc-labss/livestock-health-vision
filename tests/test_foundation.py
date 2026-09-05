"""Foundation: package layout, dependency set, configuration digest, species profile."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

STAGE_MODULES = [
    "lhv.ingest",
    "lhv.perception",
    "lhv.identity",
    "lhv.phenotype",
    "lhv.baseline",
    "lhv.events",
    "lhv.evaluation",
]


# -- 1.1 package layout -----------------------------------------------------


@pytest.mark.parametrize("module_name", STAGE_MODULES)
def test_every_stage_boundary_has_a_module(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module.__doc__, f"{module_name} does not say which stage boundary it owns"


def test_stage_modules_import_from_a_clean_interpreter() -> None:
    """Importing a stage must not depend on anything already imported in-process."""
    imports = "; ".join(f"import {name}" for name in STAGE_MODULES)
    result = subprocess.run(
        [sys.executable, "-c", f"{imports}; import lhv; print(lhv.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


# -- 1.2 dependency set -----------------------------------------------------


@pytest.mark.parametrize(
    ("module_name", "role"),
    [
        ("torch", "deep-learning runtime"),
        ("ultralytics", "pretrained perception models"),
        ("cv2", "video decoding"),
        ("pyarrow", "columnar storage"),
        ("duckdb", "embedded analytical database"),
        ("numpy", "numerics"),
        ("scipy", "signal processing"),
        ("yaml", "declarative profiles and registrations"),
    ],
)
def test_dependency_imports(module_name: str, role: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None, f"{role} dependency {module_name} failed to import"


def test_dependency_set_resolves_without_conflicts() -> None:
    """Every declared requirement, and every requirement of those, must be satisfied."""
    from importlib.metadata import PackageNotFoundError, distributions, version

    from packaging.requirements import Requirement

    installed = {d.metadata["Name"].lower(): d for d in distributions() if d.metadata["Name"]}
    conflicts: list[str] = []
    for dist in installed.values():
        for raw in dist.requires or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                found = version(requirement.name)
            except PackageNotFoundError:
                continue  # an unrequested extra, not a conflict
            if requirement.specifier and not requirement.specifier.contains(
                found, prereleases=True
            ):
                conflicts.append(
                    f"{dist.metadata['Name']} requires {raw!r} but {requirement.name} "
                    f"{found} is installed"
                )
    assert not conflicts, "\n".join(conflicts)


# -- 1.3 resolved configuration and its digest ------------------------------


def _config(**overrides):
    from lhv.config import ModelIdentity, ResolvedConfig

    base = dict(
        species_profile="cattle",
        species_profile_version="0",
        dataset_name="cattleeyeview",
        dataset_version="1.0",
        models={
            "detector": ModelIdentity(name="yolo11m", version="8.4", task="detect"),
            "pose": ModelIdentity(name="yolo11m-pose", version="8.4", task="pose"),
        },
    )
    base.update(overrides)
    return ResolvedConfig(**base)


def test_digest_is_stable_for_an_identical_configuration() -> None:
    assert _config().digest == _config().digest


def test_reordered_but_equivalent_configuration_has_the_same_digest() -> None:
    """Ordering of a mapping is not material, so it must not move the digest."""
    from lhv.config import ModelIdentity

    forward = _config(
        models={
            "detector": ModelIdentity(name="yolo11m", version="8.4", task="detect"),
            "pose": ModelIdentity(name="yolo11m-pose", version="8.4", task="pose"),
        }
    )
    reversed_order = _config(
        models={
            "pose": ModelIdentity(name="yolo11m-pose", version="8.4", task="pose"),
            "detector": ModelIdentity(name="yolo11m", version="8.4", task="detect"),
        }
    )
    assert list(forward.models) != list(reversed_order.models)
    assert forward.digest == reversed_order.digest


@pytest.mark.parametrize(
    "change",
    [
        {"dataset_version": "1.1"},
        {"species_profile_version": "1"},
        {"seed": 1},
        {"feature_set_version": "1"},
        {"event_schema_version": "2"},
    ],
)
def test_material_top_level_change_moves_the_digest(change: dict) -> None:
    assert _config().digest != _config(**change).digest


@pytest.mark.parametrize(
    ("section", "field_name", "value"),
    [
        ("ingest", "frame_stride", 2),
        ("ingest", "unreliable_timestamp_policy", "exclude"),
        ("perception", "detection_threshold", 0.30),
        ("perception", "max_track_gap_frames", 30),
        ("identity", "confidence_floor", 0.7),
        ("phenotype", "entry_boundary", 0.25),
        ("baseline", "lookback_days", 28),
        ("baseline", "min_observations", 6),
        ("events", "alert_threshold", 2.5),
        ("events", "threshold_policy_version", "2"),
        ("evaluation", "test_fraction", 0.4),
    ],
)
def test_material_section_change_moves_the_digest(section, field_name, value) -> None:
    import dataclasses

    base = _config()
    changed = dataclasses.replace(
        base, **{section: dataclasses.replace(getattr(base, section), **{field_name: value})}
    )
    assert base.digest != changed.digest, f"{section}.{field_name} must be material"


def test_model_identity_change_moves_the_digest() -> None:
    from lhv.config import ModelIdentity

    swapped = _config(
        models={
            "detector": ModelIdentity(name="yolo11x", version="8.4", task="detect"),
            "pose": ModelIdentity(name="yolo11m-pose", version="8.4", task="pose"),
        }
    )
    assert _config().digest != swapped.digest


@pytest.mark.parametrize(
    ("section", "field_name", "value"),
    [
        ("run", "output_root", "/somewhere/else"),
        ("run", "workers", 8),
        ("run", "log_level", "DEBUG"),
        ("perception", "device", "cpu"),
        ("perception", "batch_size", 32),
    ],
)
def test_immaterial_change_leaves_the_digest_alone(section, field_name, value) -> None:
    """Where a run writes and what it runs on changes its cost, not its result."""
    import dataclasses

    base = _config()
    changed = dataclasses.replace(
        base, **{section: dataclasses.replace(getattr(base, section), **{field_name: value})}
    )
    assert base.digest == changed.digest, f"{section}.{field_name} must not be material"


def test_configuration_round_trips_through_yaml(tmp_path) -> None:
    import yaml

    from lhv.config import ResolvedConfig

    original = _config()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(original.to_dict()), encoding="utf-8")
    assert ResolvedConfig.from_yaml(path).digest == original.digest


# -- 1.4 species profile ----------------------------------------------------


def test_profile_loads_and_reports_its_version() -> None:
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    assert profile.species == "cattle"
    assert profile.version
    assert profile.identifier == f"{profile.species}@{profile.version}"


def test_profile_carries_a_versioned_skeleton() -> None:
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    assert skeleton.identifier and skeleton.version
    assert len(skeleton) == 24, "CattleEyeView annotates 24 keypoints"
    assert skeleton.index_of("withers") >= 0
    assert len(set(skeleton.names)) == len(skeleton), "keypoint names must be unique"
    assert sorted(k.index for k in skeleton.keypoints) == list(range(len(skeleton)))
    for a, b in skeleton.links:
        assert a in skeleton.names and b in skeleton.names


def test_profile_carries_weight_references_with_licences() -> None:
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    assert {"detector", "pose"} <= set(profile.weights)
    for reference in profile.weights.values():
        assert reference.licence, f"{reference.name} has no recorded licence"
        assert reference.commercial_use, f"{reference.name} has no recorded commercial-use term"
    assert profile.unlicensed_weights() == ()


def test_profile_carries_a_versioned_feature_set_with_units() -> None:
    from lhv.profiles import load_profile

    feature_set = load_profile("cattle").feature_set
    assert feature_set.version == "0"
    assert len(feature_set) > 0
    for feature in feature_set.features:
        assert feature.unit, f"{feature.name} declares no unit"
        assert feature.depends_on, f"{feature.name} declares no keypoint dependency"


def test_profile_features_only_depend_on_skeleton_keypoints() -> None:
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    for feature in profile.feature_set.features:
        for dependency in feature.depends_on:
            assert dependency in profile.skeleton.names


def test_profile_carries_priors_and_a_scoring_scale() -> None:
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    assert profile.scoring_scale is not None
    assert profile.scoring_scale.maximum > profile.scoring_scale.minimum
    for feature in profile.feature_set.features:
        prior = profile.prior(feature.name)
        assert prior is not None, f"{feature.name} has no normal-range prior"
        assert prior.low < prior.high


def test_profile_rejects_a_feature_depending_on_an_absent_keypoint() -> None:
    from lhv.errors import ProfileError
    from lhv.profiles import profile_from_dict

    with pytest.raises(ProfileError, match="udder"):
        profile_from_dict(
            {
                "species": "test",
                "version": "0",
                "skeleton": {
                    "identifier": "t",
                    "version": "1",
                    "keypoints": [{"name": "withers", "index": 0}],
                },
                "weights": {},
                "feature_set": {
                    "version": "0",
                    "features": [
                        {"name": "f", "unit": "ratio", "depends_on": ["withers", "udder"]}
                    ],
                },
            }
        )


def test_missing_profile_names_what_is_available() -> None:
    from lhv.errors import ProfileError
    from lhv.profiles import load_profile

    with pytest.raises(ProfileError, match="cattle"):
        load_profile("sheep")


# -- 1.5 lint and test harness ----------------------------------------------


def test_species_seam_check_passes_on_the_package() -> None:
    from tools.check_species_seam import PACKAGE_ROOT, check_file

    findings = [
        f for path in sorted(PACKAGE_ROOT.rglob("*.py")) for f in check_file(path, PACKAGE_ROOT)
    ]
    assert not findings, "\n".join(str(f) for f in findings)


def test_species_seam_check_catches_a_hard_coded_species(tmp_path) -> None:
    from tools.check_species_seam import check_file

    module = tmp_path / "stage.py"
    module.write_text(
        "def load():\n    return {'species': 'bovine'}\n",
        encoding="utf-8",
    )
    findings = check_file(module, tmp_path)
    assert findings, "a hard-coded species name outside the profile must fail the check"
    assert "bovine" in str(findings[0])


def test_species_seam_check_catches_a_branch_on_species(tmp_path) -> None:
    from tools.check_species_seam import check_file

    module = tmp_path / "stage.py"
    module.write_text(
        "def stride(profile):\n    if profile.species == 'x':\n        return 1\n    return 2\n",
        encoding="utf-8",
    )
    findings = check_file(module, tmp_path)
    assert any(f.kind == "branch on species" for f in findings)


def test_species_seam_check_exempts_the_profile_package(tmp_path) -> None:
    from tools.check_species_seam import check_file

    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    module = profile_dir / "definitions.py"
    module.write_text("DEFAULT = 'bovine'\n", encoding="utf-8")
    assert check_file(module, tmp_path) == []


def test_ci_runs_the_check_harness() -> None:
    """The check harness must actually run in CI, not just exist on disk."""
    from pathlib import Path

    workflow = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    assert workflow.exists(), "no CI workflow"
    text = workflow.read_text(encoding="utf-8")
    assert "tools/check.sh" in text
    harness = Path(__file__).resolve().parent.parent / "tools" / "check.sh"
    assert "check_species_seam.py" in harness.read_text(encoding="utf-8")


def test_the_default_profile_is_resolved_from_what_is_installed() -> None:
    """No caller outside the profile package may name a species, including a default."""
    from lhv.profiles import available_profiles, default_profile_name

    assert default_profile_name() in available_profiles()


def test_the_cli_does_not_hard_code_a_species() -> None:
    from lhv.cli import build_parser

    parser = build_parser()
    assert parser.parse_args(["profiles", "show"]).name is None
    assert (
        parser.parse_args(["run", "--dataset", "d", "--data-root", "r", "--output", "o"]).profile
        is None
    )


# -- the skeleton matches the release it claims to follow --------------------

# Verbatim from CattleEyeView's own cattleeyeview_pose.yaml (list_keypoints).
RELEASE_KEYPOINT_ORDER = [
    "head",
    "nose",
    "eyeL",
    "eyeR",
    "earbaseL",
    "eartipL",
    "earbaseR",
    "eartipR",
    "neck",
    "withers",
    "elbowFL",
    "kneeFL",
    "pawFL",
    "elbowFR",
    "kneeFR",
    "pawFR",
    "elbowBL",
    "kneeBL",
    "pawBL",
    "elbowBR",
    "kneeBR",
    "pawBR",
    "tailbase",
    "tailend",
]
# Verbatim from the release's cattleeyeview_pose.yaml (flip_idx).
RELEASE_FLIP_INDEX = [
    0,
    1,
    3,
    2,
    6,
    7,
    4,
    5,
    8,
    9,
    13,
    14,
    15,
    10,
    11,
    12,
    19,
    20,
    21,
    16,
    17,
    18,
    22,
    23,
]


def test_the_skeleton_index_order_matches_the_release() -> None:
    """The profile claims to follow CattleEyeView; this is that claim, checked."""
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    assert [k.alias for k in skeleton.by_index()] == RELEASE_KEYPOINT_ORDER


def test_the_flip_pairs_match_the_release() -> None:
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    ordered = skeleton.by_index()
    by_name = {k.name: k for k in ordered}

    for index, keypoint in enumerate(ordered):
        mirrored = RELEASE_FLIP_INDEX[index]
        expected = ordered[mirrored].name
        if mirrored == index:
            assert keypoint.swap == "", f"{keypoint.name} has no mirror in the release"
        else:
            assert keypoint.swap == expected
            assert by_name[keypoint.swap].swap == keypoint.name, "flips must be reciprocal"


def test_every_keypoint_carries_the_published_oks_sigma() -> None:
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    assert set(skeleton.sigmas) == {0.025}
    assert len(skeleton.sigmas) == 24


def test_the_release_names_map_onto_the_profile_names() -> None:
    """A label file written in the release's names must be readable."""
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    aliases = skeleton.aliases
    assert len(aliases) == 24
    assert aliases["pawFL"] == "left_front_paw"
    assert aliases["tailbase"] == "base_of_tail"
    assert set(aliases.values()) == set(skeleton.names)


def test_the_links_match_the_release_topology() -> None:
    """Every limb attaches to the withers; the head has no link to the neck."""
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    links = {tuple(link) for link in skeleton.links}
    assert len(skeleton.links) == 22
    for limb in (
        "left_front_elbow",
        "right_front_elbow",
        "left_back_elbow",
        "right_back_elbow",
    ):
        assert ("withers", limb) in links
    assert ("withers", "base_of_tail") in links
    assert ("head", "neck") not in links
    assert ("neck", "withers") in links
