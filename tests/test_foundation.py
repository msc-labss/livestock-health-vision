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
    assert len(skeleton) == 9, "feature-set version 1 needs nine points and no more"
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
    assert feature_set.version == "1"
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


def test_the_skeleton_names_the_convention_it_is_a_subset_of() -> None:
    """Version 1's point set is CoWalk-17 minus the intermediate limb joints.

    It was chosen from the feature set before the CoWalk count was known, and
    turned out to carry that convention's five body landmarks exactly. The
    source field must name the convention and the omission rather than claim a
    provenance it does not have or hide the one it does.
    """
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    assert skeleton.view == "lateral"
    assert "CoWalk" in skeleton.source
    assert "arXiv:2104.08029" in skeleton.source
    # It stays provisional: whether the omitted joints carry information this
    # feature set is missing is a question for the pilot, not the literature.
    assert skeleton.provisional is True


def test_no_keypoint_carries_an_invented_oks_sigma() -> None:
    """No published sigma exists for this point set, so none is recorded."""
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    assert skeleton.sigma_source == "none-established"
    assert set(skeleton.sigmas) == {0.0}, "a sigma with no provenance is worse than none"


def test_the_flip_pairs_are_reciprocal_and_cover_the_hooves() -> None:
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    by_name = {k.name: k for k in skeleton.keypoints}

    hooves = [n for n in skeleton.names if n.endswith("_hoof")]
    assert len(hooves) == 4
    for name in hooves:
        swap = by_name[name].swap
        assert swap, f"{name} declares no mirror"
        assert by_name[swap].swap == name, "flips must be reciprocal"

    # The dorsal line and head sit on the midline and mirror onto themselves.
    for name in ("nose", "forehead", "withers", "mid_thoracic", "sacrum"):
        assert by_name[name].swap == "", f"{name} is on the midline and needs no mirror"


def test_the_skeleton_carries_exactly_what_the_feature_set_depends_on() -> None:
    """The point set follows the features, not a convention chosen before them."""
    from lhv.profiles import load_profile

    profile = load_profile("cattle")
    depended_on = {d for f in profile.feature_set.features for d in f.depends_on}
    names = set(profile.skeleton.names)

    assert depended_on <= names, "a feature depends on a keypoint the skeleton lacks"
    # forehead is the one point no version-1 feature uses: it is carried for the
    # head-pitch measurement a later feature set is expected to want, and the
    # test states that rather than letting it look accidental.
    assert names - depended_on == {"forehead"}


def test_the_links_span_the_dorsal_line_and_reach_every_hoof() -> None:
    from lhv.profiles import load_profile

    skeleton = load_profile("cattle").skeleton
    links = {tuple(link) for link in skeleton.links}

    assert ("withers", "mid_thoracic") in links
    assert ("mid_thoracic", "sacrum") in links
    for hoof in ("left_front_hoof", "right_front_hoof"):
        assert ("withers", hoof) in links
    for hoof in ("left_hind_hoof", "right_hind_hoof"):
        assert ("sacrum", hoof) in links


# -- two profiles, two feature-set versions, side by side --------------------


def test_both_profiles_are_installed_and_declare_different_geometries() -> None:
    from lhv.profiles import available_profiles, load_profile

    assert set(available_profiles()) == {"cattle", "cattle-topdown"}
    lateral = load_profile("cattle")
    topdown = load_profile("cattle-topdown")

    assert lateral.skeleton.view == "lateral"
    assert topdown.skeleton.view == "top-down"
    assert lateral.skeleton.identifier != topdown.skeleton.identifier
    assert lateral.feature_set.version != topdown.feature_set.version


def test_the_default_profile_is_declared_rather_than_inferred() -> None:
    """With more than one installed, being the only file no longer identifies it."""
    from lhv.profiles import load_profile
    from lhv.profiles.profile import default_profile_name

    assert default_profile_name() == "cattle"
    assert load_profile(default_profile_name()).skeleton.view == "lateral"


def test_records_from_the_two_versions_are_distinguishable_by_their_version_alone() -> None:
    """Version 0 records stay readable and are not migrated."""
    from lhv.profiles import load_profile

    versions = {load_profile(n).feature_set.version for n in ("cattle", "cattle-topdown")}
    assert versions == {"0", "1"}, "the two versions must not collide"

    # The names genuinely differ, so a reader cannot confuse a v0 column for a v1
    # one even before consulting the version field.
    lateral = set(load_profile("cattle").feature_set.names)
    topdown = set(load_profile("cattle-topdown").feature_set.names)
    assert lateral & topdown == {"speed"}


def test_every_available_feature_in_both_profiles_has_an_implementation() -> None:
    """A profile may not declare a feature available that nothing can compute."""
    from lhv.phenotype.features import _IMPLEMENTATIONS
    from lhv.profiles import load_profile

    for name in ("cattle", "cattle-topdown"):
        for feature in load_profile(name).feature_set.available:
            assert feature.name in _IMPLEMENTATIONS, (
                f"{name} declares {feature.name} available with no implementation"
            )


# -- a profile may not declare weights nothing honours -----------------------


def test_every_declared_weight_is_wired_or_says_why_not() -> None:
    """The check that enforces it, run as a test so it cannot be skipped.

    Three declarations in this profile have outlived the thing they referred to:
    the skeleton's view, the pose runtime, and the re-identification weights.
    None was caught by a test, because tests exercise what the code does and
    these were claims about what it does.
    """
    import subprocess
    import sys
    from pathlib import Path as _Path

    root = _Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "tools" / "check_profile_declarations.py")],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_the_reid_weights_are_recorded_as_not_wired() -> None:
    """They were declared as "used only by the visual fallback" and used by nothing.

    The fallback runs a weights-free colour histogram, so the figures P0 reports
    for visual re-identification came from that histogram rather than from these
    weights. The entry now says so.
    """
    from lhv.profiles import available_profiles, load_profile

    for name in available_profiles():
        reference = load_profile(name).weight("reid")
        assert reference.not_wired.strip(), f"{name} claims its reid weights are wired"
        assert "histogram" in reference.not_wired


def test_the_lateral_profiles_pose_records_why_it_is_not_wired() -> None:
    from lhv.profiles import load_profile

    reference = load_profile("cattle").weight("pose")
    assert reference.runtime == "mmpose"
    assert "not implemented" in reference.not_wired
