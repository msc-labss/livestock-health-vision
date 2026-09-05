#!/usr/bin/env python3
"""Verify and unpack the MultiCamCows2024 archive.

The archive is 36.6 GB, of which 33 GB is raw monitor footage and 3.4 GB is the
tracklet imagery carrying the identity labels. Which components are unpacked is
therefore a choice, and it is made explicitly here rather than by whatever
happens to fit on the volume.

Byte length is checked against the server before anything is unpacked, because
a truncated archive that unpacks partially is worse than one that fails.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

DOWNLOAD_URL = "https://data.bris.ac.uk/datasets/tar/2inu67jru7a6821kkgehxg3cv2.zip"
ARCHIVE_PREFIX = "2inu67jru7a6821kkgehxg3cv2/"


def remote_size(url: str) -> int | None:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return int(response.headers["Content-Length"])
    except Exception as exc:  # network trouble is not a verification failure
        print(f"could not ask the server for the archive length: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--into", type=Path, required=True)
    parser.add_argument(
        "--component",
        default="images",
        choices=["images", "videos", "all"],
        help="images: the tracklet imagery and metadata; videos: the monitor footage",
    )
    parser.add_argument("--skip-size-check", action="store_true")
    args = parser.parse_args()

    if not args.archive.exists():
        print(f"no archive at {args.archive}", file=sys.stderr)
        return 1

    local = args.archive.stat().st_size
    print(f"archive: {local:,} bytes ({local / 2**30:.2f} GiB)")
    if not args.skip_size_check:
        expected = remote_size(DOWNLOAD_URL)
        if expected is not None:
            if local != expected:
                print(
                    f"INCOMPLETE: the server reports {expected:,} bytes, "
                    f"{expected - local:,} still missing. Nothing unpacked.",
                    file=sys.stderr,
                )
                return 1
            print(f"byte length matches the server ({expected:,})")

    with zipfile.ZipFile(args.archive) as archive:
        names = archive.namelist()
        wanted = [n for n in names if _wanted(n, args.component)]
        total = sum(archive.getinfo(n).file_size for n in wanted)
        print(
            f"unpacking {len(wanted):,} of {len(names):,} entries "
            f"({total / 2**30:.2f} GiB) into {args.into}"
        )

        free = _free_bytes(args.into)
        if free is not None and total > free * 0.95:
            print(
                f"REFUSING: {total / 2**30:.2f} GiB needed, {free / 2**30:.2f} GiB free.",
                file=sys.stderr,
            )
            return 1

        args.into.mkdir(parents=True, exist_ok=True)
        done = 0
        for name in wanted:
            info = archive.getinfo(name)
            relative = name[len(ARCHIVE_PREFIX) :] if name.startswith(ARCHIVE_PREFIX) else name
            if not relative or relative.endswith("/"):
                (args.into / relative).mkdir(parents=True, exist_ok=True)
                continue
            target = args.into / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as sink:
                while chunk := source.read(1 << 20):
                    sink.write(chunk)
            done += 1
            if done % 5000 == 0:
                print(f"  {done:,}/{len(wanted):,}")

    print(f"unpacked {done:,} file(s) into {args.into}")
    return 0


def _wanted(name: str, component: str) -> bool:
    relative = name[len(ARCHIVE_PREFIX) :] if name.startswith(ARCHIVE_PREFIX) else name
    if component == "all":
        return True
    is_video = relative.startswith("MultiCamCows2024Root/videos")
    if component == "videos":
        return is_video
    return not is_video


def _free_bytes(path: Path) -> int | None:
    import shutil

    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
