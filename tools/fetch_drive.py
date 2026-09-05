#!/usr/bin/env python3
"""List and download a public Google Drive folder.

Written because the datasets this repository uses arrive that way, and because
Drive fails in a shape that quietly corrupts a download: when a folder passes
its per-day anonymous quota, every file returns HTTP 200 carrying an HTML page
titled "Quota exceeded". A downloader that trusts the status code writes that
page to disk under the file's name, and the failure surfaces much later as an
unreadable video.

So every response body is inspected before anything is written, and a file whose
content looks like Drive's HTML rather than the file it claims to be is refused
by name.

    python tools/fetch_drive.py list   <folder-id>
    python tools/fetch_drive.py fetch  <folder-id> --into data/<name>
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FOLDER_MIME = "application/vnd.google-apps.folder"
USER_AGENT = "Mozilla/5.0"
_LISTING = re.compile(r"_DRIVE_ivd'\]\s*=\s*'(.*?)';", re.S)
_ENTRY = re.compile(r'\["([\w-]{20,})",\["([\w-]{20,})"\],"([^"]+)","([^"]+)",(\d*)')


class QuotaExceeded(RuntimeError):
    """Drive refused the download in a 200 response. Nothing may be written."""


@dataclass(frozen=True)
class Entry:
    id: str
    name: str
    mime: str
    size: int

    @property
    def is_folder(self) -> bool:
        return self.mime == FOLDER_MIME


def _get(url: str, timeout: int = 120) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return response.read()


def list_folder(folder_id: str) -> list[Entry]:
    """Entries directly inside a public folder."""
    html = _get(f"https://drive.google.com/drive/folders/{folder_id}").decode("utf-8", "replace")
    found = _LISTING.search(html)
    if not found:
        raise RuntimeError(f"folder {folder_id} returned no listing; it may be private or renamed")
    # Drive escapes forward slashes as \/, which unicode_escape does not know
    # and warns about. Undo those first so only real escapes remain.
    escaped = found.group(1).replace("\\/", "/")
    decoded = escaped.encode().decode("unicode_escape", "replace")
    entries = []
    for file_id, _parent, name, mime, size in _ENTRY.findall(decoded):
        entries.append(
            Entry(
                id=file_id,
                name=name.replace("\\/", "/"),
                mime=mime.replace("\\/", "/"),
                size=int(size) if size else 0,
            )
        )
    return sorted(entries, key=lambda e: (not e.is_folder, e.name))


def looks_like_drive_error(payload: bytes) -> str:
    """Name the Drive interstitial hiding inside a 200 response, if there is one."""
    head = payload[:4096].lower()
    if b"<!doctype html" not in head and b"<html" not in head:
        return ""
    if b"quota exceeded" in head:
        return "Drive returned its 'Quota exceeded' page instead of the file"
    if b"sign in" in head or b"accountchooser" in head:
        return "Drive returned a sign-in page; the file is not publicly readable"
    if b"virus scan warning" in head or b"can't scan" in head:
        return "Drive returned its virus-scan interstitial"
    return "Drive returned an HTML page instead of the file"


def download(entry: Entry, target: Path, *, timeout: int = 600) -> Path:
    url = f"https://drive.usercontent.google.com/download?id={entry.id}&export=download&confirm=t"
    payload = _get(url, timeout=timeout)
    problem = looks_like_drive_error(payload)
    if problem:
        raise QuotaExceeded(f"{entry.name}: {problem}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    return target


def walk(folder_id: str, prefix: str = "", depth: int = 0, maxdepth: int = 6):
    for entry in list_folder(folder_id):
        path = f"{prefix}/{entry.name}" if prefix else entry.name
        yield entry, path
        if entry.is_folder and depth < maxdepth:
            yield from walk(entry.id, path, depth + 1, maxdepth)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["list", "fetch"])
    parser.add_argument("folder_id")
    parser.add_argument("--into", type=Path, default=None)
    parser.add_argument("--maxdepth", type=int, default=6)
    args = parser.parse_args()

    if args.action == "list":
        for entry, path in walk(args.folder_id, maxdepth=args.maxdepth):
            kind = "DIR " if entry.is_folder else "FILE"
            print(f"{kind} {entry.size:>14,}  {path}")
        return 0

    if args.into is None:
        print("--into is required for fetch", file=sys.stderr)
        return 2

    refused, written, skipped = [], 0, 0
    for entry, path in walk(args.folder_id, maxdepth=args.maxdepth):
        if entry.is_folder:
            continue
        target = args.into / path
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        try:
            download(entry, target)
        except QuotaExceeded as exc:
            refused.append(str(exc))
            continue
        written += 1
        print(f"  {target} ({target.stat().st_size:,} bytes)")

    print(f"\n{written} file(s) written, {skipped} already present")
    if refused:
        print(f"{len(refused)} refused, nothing written for them:", file=sys.stderr)
        for line in refused[:10]:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nDrive's anonymous quota resets after about a day. Copying the folder into "
            "your own Drive gives it a fresh quota.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
