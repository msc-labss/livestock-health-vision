#!/usr/bin/env python3
"""Fail the build if a profile declares weights nothing honours.

The species profile is where decisions get written down, and three times now a
declaration has outlived the thing it referred to. The skeleton's view was
parsed and consulted by nothing. The pose weights moved to a checkpoint format
no implemented backend can read. The re-identification weights were declared as
"used only by the visual fallback" while nothing loaded them and the fallback
ran a weights-free histogram.

None of those was caught by a test, because tests exercise what the code does
and these were claims about what it does. This check is the claim, enforced.

The invariant: every weight role a profile declares is either built by a stage
from that profile under a runtime the stage implements, or carries `not_wired`
saying why it is not. Recording an intended backend before one exists is
legitimate and is how a decision gets written down; leaving it indistinguishable
from one that works is not.

Run with no arguments to check every installed profile:

    python tools/check_profile_declarations.py
"""

from __future__ import annotations

import importlib
import sys

from lhv.profiles import available_profiles, load_profile

# Role -> the stage function that builds it from a profile, or None where no
# stage does. Update this when a builder is added; the check verifies that every
# function named here actually imports, so a stale entry fails rather than
# quietly excusing a role.
BUILDERS: dict[str, str | None] = {
    "detector": "lhv.perception.detect:detector_from_profile",
    "pose": "lhv.perception.pose:pose_backend_from_profile",
    # Nothing builds an embedding from the profile. The fallback constructs a
    # weights-free colour histogram directly, so this reference is unreachable
    # and must say so.
    "reid": None,
}

# Role -> the runtimes its builder can actually load.
IMPLEMENTED: dict[str, tuple[str, ...]] = {
    "detector": ("ultralytics",),
    "pose": ("ultralytics",),
    "reid": (),
}


def _builder_exists(target: str) -> bool:
    module_name, _, attribute = target.partition(":")
    try:
        return hasattr(importlib.import_module(module_name), attribute)
    except ImportError:
        return False


def main() -> int:
    problems: list[str] = []
    checked = 0

    for name in BUILDERS.values():
        if name and not _builder_exists(name):
            problems.append(f"BUILDERS names {name!r}, which does not import; the map is stale")

    for profile_name in available_profiles():
        profile = load_profile(profile_name)
        for role, reference in sorted(profile.weights.items()):
            checked += 1
            where = f"{profile_name}.weights.{role}"

            if role not in BUILDERS:
                problems.append(
                    f"{where}: unknown role. Add it to BUILDERS and IMPLEMENTED in this "
                    f"check, so a new role cannot arrive unexamined"
                )
                continue

            buildable = BUILDERS[role] is not None and reference.runtime in IMPLEMENTED[role]
            if buildable and reference.not_wired:
                problems.append(
                    f"{where}: declares not_wired, but {role} is built from the profile and "
                    f"{reference.runtime!r} is implemented. Remove the reason or the entry is "
                    f"understating what works"
                )
            elif not buildable and not reference.not_wired:
                if BUILDERS[role] is None:
                    why = f"no stage builds {role!r} from the profile"
                else:
                    why = (
                        f"runtime {reference.runtime!r} is not implemented for {role!r} "
                        f"(implemented: {', '.join(IMPLEMENTED[role]) or 'none'})"
                    )
                problems.append(
                    f"{where}: {why}, and the entry does not say so. Add 'not_wired' with the "
                    f"reason, or wire it. A declaration nobody honours reads like one that works"
                )

    if problems:
        print("profile declarations are not honoured:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print(f"profile declarations hold: {checked} weight reference(s) checked, each wired or stated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
