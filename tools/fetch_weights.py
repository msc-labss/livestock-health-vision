#!/usr/bin/env python3
"""Fetch the pretrained weights the species profile names.

Weights are never committed. This script puts them where the profile expects
them, and prints the licence recorded against each, because a weight file
arriving without its licence in view is how a licensing problem becomes a
surprise later.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lhv.profiles import load_profile  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="cattle")
    parser.add_argument("--into", default="weights")
    args = parser.parse_args()

    profile = load_profile(args.profile)
    destination = Path(args.into)
    destination.mkdir(parents=True, exist_ok=True)

    for role, reference in sorted(profile.weights.items()):
        if not reference.uri.startswith("http"):
            print(f"{role}: {reference.name} is not a downloadable file ({reference.uri}); skipped")
            continue
        target = destination / Path(reference.uri).name
        print(f"{role}: {reference.name}@{reference.version}  [{reference.licence}]")
        print(f"       commercial use: {reference.commercial_use}")
        if target.exists():
            print(f"       already present at {target}")
            continue
        print(f"       downloading {reference.uri} -> {target}")
        urllib.request.urlretrieve(reference.uri, target)  # noqa: S310

    unlicensed = profile.unlicensed_weights()
    if unlicensed:
        print("\nWARNING: weights with no recorded licence:", file=sys.stderr)
        for reference in unlicensed:
            print(f"  {reference.role}: {reference.name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
