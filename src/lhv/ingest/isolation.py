"""Derived-media isolation.

Decoded frames, evidence clips and every other derived medium are written only
beneath roots that version control ignores. The check runs before anything is
written, because a path discovered after the write is a leak that already
happened.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..errors import DerivedMediaIsolationError

__all__ = ["assert_isolated_output_root", "is_tracked_by_version_control"]


def _git_root(path: Path) -> Path | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - git absent
        return None
    if result.returncode != 0:
        return None
    return Path(result.stdout.strip())


def is_tracked_by_version_control(path: Path) -> tuple[bool, str]:
    """Report whether version control would pick up writes beneath ``path``.

    A path is unsafe when it lies inside a repository and is not ignored. The
    question asked of git is deliberately "would you track a file written here",
    not "does this path exist", so an empty directory is judged correctly.
    """
    path = Path(path).resolve()
    search_from = path
    while not search_from.exists() and search_from != search_from.parent:
        search_from = search_from.parent

    root = _git_root(search_from)
    if root is None:
        return False, "not inside a version-controlled tree"

    probe = path / ".lhv-write-probe"
    result = subprocess.run(
        ["git", "-C", str(root), "check-ignore", "-q", str(probe)],
        capture_output=True,
        text=True,
        check=False,
    )
    # check-ignore exits 0 when the path IS ignored, 1 when it is not.
    if result.returncode == 0:
        return False, f"ignored by version control in {root}"
    return True, f"tracked by version control in {root}"


def assert_isolated_output_root(path: str | Path) -> Path:
    """Refuse an output root that version control would track. Called before writing."""
    resolved = Path(path).resolve()
    tracked, reason = is_tracked_by_version_control(resolved)
    if tracked:
        raise DerivedMediaIsolationError(
            f"refusing to write derived media to {resolved}: {reason}. Configure an output "
            f"root that version control ignores.",
            path=str(resolved),
        )
    return resolved
