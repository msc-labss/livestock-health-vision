"""Command line entry point.

The P0 gate is "the pipeline runs end to end with no manual step between
stages". That is a property of this file: one invocation carries a registered
dataset from source registration to exported events, and the stage subcommands
exist so a later stage can be re-run without the earlier ones, not so a person
has to run them in order.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ResolvedConfig
from .datasets import available_registrations, load_registration
from .ingest.source import register_sources_from_dataset
from .pipeline import Pipeline
from .profiles import available_profiles, default_profile_name, load_profile

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lhv", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    datasets = subparsers.add_parser("datasets", help="registered datasets and their terms")
    datasets.add_argument("action", choices=["list", "verify", "show"])
    datasets.add_argument("name", nargs="?")
    datasets.add_argument("--data-root", default=None)

    profiles = subparsers.add_parser("profiles", help="species profiles")
    profiles.add_argument("action", choices=["list", "show"])
    profiles.add_argument("name", nargs="?", default=None)

    run = subparsers.add_parser("run", help="run the pipeline end to end")
    run.add_argument("--dataset", required=True)
    run.add_argument("--data-root", required=True)
    run.add_argument("--output", required=True)
    run.add_argument("--config", default=None)
    run.add_argument(
        "--profile", default=None, help="species profile; defaults to the only one installed"
    )
    run.add_argument("--weights-root", default="weights")
    run.add_argument(
        "--stages",
        default="all",
        help=(
            "comma-separated stages to run: perception, identity, phenotype, baseline, "
            "events; or 'all'. Later stages read the records earlier ones wrote."
        ),
    )
    run.add_argument("--frame-width", type=int, default=0)
    run.add_argument("--frame-height", type=int, default=0)

    evaluate = subparsers.add_parser(
        "evaluate", help="build the evaluation report from a run's records"
    )
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--profile", default=None)
    evaluate.add_argument("--config", default=None)
    evaluate.add_argument("--split", default="animal", choices=["frame", "animal", "day", "site"])
    evaluate.add_argument("--into", default=None, help="write the report here as well")

    check = subparsers.add_parser(
        "recording-check", help="judge a pilot recording against the P1 specification"
    )
    check.add_argument("video")
    check.add_argument("--profile", default=None)
    check.add_argument("--weights-root", default="weights")
    check.add_argument("--seconds", type=float, default=60.0)
    check.add_argument(
        "--no-detector",
        action="store_true",
        help="skip everything that needs to look at the frames",
    )

    report = subparsers.add_parser("report", help="print a run's recorded outcome")
    report.add_argument("--output", required=True)

    return parser


def _load_config(path: str | None, *, profile_name: str, dataset) -> ResolvedConfig:
    if path:
        return ResolvedConfig.from_yaml(path)
    profile = load_profile(profile_name)
    from .config import ModelIdentity

    return ResolvedConfig(
        species_profile=profile.species,
        species_profile_version=profile.version,
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        models={
            role: ModelIdentity(
                name=reference.name,
                version=reference.version,
                task=role,
                weights_uri=reference.uri,
                licence=reference.licence,
            )
            for role, reference in profile.weights.items()
        },
        feature_set_version=profile.feature_set.version,
    )


def _cmd_datasets(args) -> int:
    if args.action == "list":
        for name in available_registrations():
            registration = load_registration(name)
            route = registration.access.access_route
            print(f"{name}@{registration.version}  site={registration.site_key}  access={route}")
        return 0

    if not args.name:
        print("a dataset name is required", file=sys.stderr)
        return 2
    registration = load_registration(args.name)

    if args.action == "show":
        print(json.dumps(registration.to_dict(), indent=2, default=str))
        return 0

    report = registration.verify(args.data_root)
    print(report.describe())
    if not report.present:
        access = registration.access
        print(
            f"\nAccess route: {access.access_route}."
            + (f" Request at {access.request_url}." if access.request_url else "")
            + (f" Download from {access.download_url}." if access.download_url else "")
        )
    return 0 if report.ok else 1


def _cmd_profiles(args) -> int:
    if args.action == "list":
        for name in available_profiles():
            profile = load_profile(name)
            print(
                f"{profile.identifier}  skeleton={profile.skeleton.identifier}"
                f"@{profile.skeleton.version}  features=v{profile.feature_set.version}"
            )
        return 0

    profile = load_profile(args.name or default_profile_name())
    print(f"{profile.identifier}")
    print(
        f"  skeleton: {profile.skeleton.identifier}@{profile.skeleton.version} "
        f"({len(profile.skeleton)} keypoints, {profile.skeleton.view})"
    )
    print(f"  feature set: v{profile.feature_set.version} ({len(profile.feature_set)} features)")
    for feature in profile.feature_set.features:
        print(f"    - {feature.name} [{feature.unit}]")
    print("  weights:")
    for role, reference in sorted(profile.weights.items()):
        print(
            f"    - {role}: {reference.name}@{reference.version} "
            f"licence={reference.licence} commercial={reference.commercial_use}"
        )
    if profile.scoring_scale:
        print(f"  scoring scale: {profile.scoring_scale.name}")
    return 0


def _cmd_run(args) -> int:
    registration = load_registration(args.dataset)
    verification = registration.verify(args.data_root)
    if not verification.present:
        print(verification.describe(), file=sys.stderr)
        print(
            f"{registration.name} is registered but not present. Access route: "
            f"{registration.access.access_route}.",
            file=sys.stderr,
        )
        return 1

    profile_name = args.profile or default_profile_name()
    profile = load_profile(profile_name)
    config = _load_config(args.config, profile_name=profile_name, dataset=registration)
    sources = register_sources_from_dataset(registration, data_root=args.data_root)
    if not sources:
        print(f"no media found beneath {args.data_root}", file=sys.stderr)
        return 1

    width, height = args.frame_width, args.frame_height
    if not width or not height:
        width, height = _probe_frame_size(sources[0])

    pipeline = Pipeline(config, profile, args.output, weights_root=args.weights_root)
    stages = (
        ["perception", "identity", "phenotype", "baseline", "events"]
        if args.stages == "all"
        else [s.strip() for s in args.stages.split(",") if s.strip()]
    )

    if "perception" in stages:
        pipeline.run_perception(sources)
    if "identity" in stages:
        pipeline.run_identity()
    if "phenotype" in stages:
        pipeline.run_phenotype(frame_width=width, frame_height=height)
    if "baseline" in stages:
        pipeline.run_baseline()
    if "events" in stages:
        pipeline.run_events(sources=sources)

    print(pipeline.result.describe())
    _write_outcome(Path(args.output), pipeline.result)
    return 0


def _probe_frame_size(source) -> tuple[int, int]:
    from .ingest.decode import open_decoder

    decoder = open_decoder(source.media_path, kind=source.kind)
    try:
        for decoded in decoder.iter_frames(decode=True):
            if decoded.image is not None:
                height, width = decoded.image.shape[:2]
                return width, height
    finally:
        decoder.close()
    return 0, 0


def _write_outcome(output: Path, result) -> Path:
    import dataclasses

    path = output / "run.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataclasses.asdict(result)
    payload["perception"] = [dataclasses.asdict(r) for r in result.perception]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _cmd_recording_check(args) -> int:
    from .config import ModelIdentity
    from .recording import Requirements, check_recording

    profile_name = args.profile or default_profile_name()
    profile = load_profile(profile_name)
    reference = profile.weight("detector")
    config = ResolvedConfig(
        species_profile=profile.species,
        species_profile_version=profile.version,
        dataset_name="pilot",
        dataset_version="0",
        models={
            "detector": ModelIdentity(name=reference.name, version=reference.version, task="detect")
        },
    )

    backend = None
    if not args.no_detector:
        from .perception.detect import UltralyticsDetector

        weights = Path(args.weights_root) / Path(reference.uri).name
        if not weights.exists():
            print(
                f"{weights} is not present; run tools/fetch_weights.py, or pass --no-detector "
                f"to check only what the container can answer",
                file=sys.stderr,
            )
            return 1
        backend = UltralyticsDetector(
            str(weights),
            name=reference.name,
            version=reference.version,
            target_classes=reference.target_classes,
            device=config.perception.device,
            confidence_threshold=config.perception.detection_threshold,
        )

    report = check_recording(
        args.video,
        profile,
        config,
        detector_backend=backend,
        requirements=Requirements(segment_seconds=args.seconds),
    )
    print(report.describe())
    return 0 if report.ok else 1


def _cmd_evaluate(args) -> int:
    from .baseline.store import TimeSeriesStore
    from .evaluation import EvaluationHarness, SplitKind
    from .stagestore import StageStore

    registration = load_registration(args.dataset)
    profile_name = args.profile or default_profile_name()
    profile = load_profile(profile_name)
    config = _load_config(args.config, profile_name=profile_name, dataset=registration)

    output = Path(args.output)
    store = StageStore(output, isolate=False)
    series = TimeSeriesStore(output / "series", profile.feature_set)

    harness = EvaluationHarness(store, series, config, profile)
    report = harness.build(
        split_kind=SplitKind(args.split),
        dataset_name=registration.name,
        dataset_version=registration.version,
    )
    rendered = report.render()
    print(rendered)

    destination = Path(args.into) if args.into else output / "evaluation-report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    destination.with_suffix(".json").write_text(report.to_json(), encoding="utf-8")
    return 0


def _cmd_report(args) -> int:
    path = Path(args.output) / "run.json"
    if not path.exists():
        print(f"no run recorded at {path}", file=sys.stderr)
        return 1
    print(path.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "datasets":
        return _cmd_datasets(args)
    if args.command == "profiles":
        return _cmd_profiles(args)
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "recording-check":
        return _cmd_recording_check(args)
    if args.command == "evaluate":
        return _cmd_evaluate(args)
    if args.command == "report":
        return _cmd_report(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
