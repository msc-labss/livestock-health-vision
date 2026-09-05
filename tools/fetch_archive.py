#!/usr/bin/env python3
"""Fetch a large archive over several connections, resumably.

Research data repositories commonly throttle a single connection well below the
available bandwidth, which turns a 36 GB download into an overnight job. Range
requests are not throttled as a group, so several segments in parallel finish in
a fraction of the time.

Segments are written into one pre-allocated file at their own offsets rather
than into part files concatenated afterwards. Concatenation would need twice the
archive's size in free space, which for the datasets this repository uses is the
difference between fitting and not.

Progress is recorded per segment, so an interrupted run resumes rather than
restarts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CHUNK = 1 << 20


def plan_segments(total: int, connections: int, *, have: int = 0) -> list[dict]:
    """Split a byte range into segments, crediting any contiguous prefix already on disk.

    An interrupted single-connection download leaves a valid prefix. Those bytes
    belong to whichever segments they cover, and re-fetching them would be pure
    waste.

    ``have`` must be a *contiguous* prefix. A segmented run pre-allocates the
    whole file, so its on-disk size says nothing about how much is valid — the
    caller is responsible for passing 0 in that case. See ``Segmented._plan``.
    """
    size = total // connections
    segments = []
    for index in range(connections):
        start = index * size
        end = total - 1 if index == connections - 1 else start + size - 1
        length = end - start + 1
        segments.append({"start": start, "end": end, "done": max(0, min(have - start, length))})
    return segments


def remote_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return int(response.headers["Content-Length"])


class Segmented:
    def __init__(self, url: str, target: Path, *, connections: int, attempts: int = 20) -> None:
        self.url = url
        self.target = target
        self.state_path = target.with_suffix(target.suffix + ".progress.json")
        self.connections = connections
        self.attempts = attempts
        self.lock = threading.Lock()
        self.total = remote_length(url)
        self.segments = self._plan()
        self.stop = threading.Event()

    def _plan(self) -> list[dict]:
        if self.state_path.exists():
            state = json.loads(self.state_path.read_text())
            if state.get("total") == self.total and len(state["segments"]) == self.connections:
                return state["segments"]

        # A file that is already the full length is either finished or merely
        # pre-allocated by an earlier segmented run whose progress record is
        # gone. Those are indistinguishable from the outside, and guessing
        # "finished" would declare a corrupt archive complete. Only a short
        # file is a trustworthy contiguous prefix.
        have = 0
        if self.target.exists():
            size = self.target.stat().st_size
            have = size if size < self.total else 0
        return plan_segments(self.total, self.connections, have=have)

    def _save(self) -> None:
        self.state_path.write_text(
            json.dumps({"total": self.total, "segments": self.segments}, indent=1)
        )

    @property
    def downloaded(self) -> int:
        return sum(s["done"] for s in self.segments)

    def _allocate(self) -> None:
        with open(self.target, "ab"):
            pass
        if self.target.stat().st_size < self.total:
            with open(self.target, "r+b") as handle:
                handle.truncate(self.total)

    def _worker(self, index: int, handle_fd: int) -> None:
        segment = self.segments[index]
        for attempt in range(self.attempts):
            length = segment["end"] - segment["start"] + 1
            if segment["done"] >= length or self.stop.is_set():
                return
            begin = segment["start"] + segment["done"]
            request = urllib.request.Request(
                self.url, headers={"Range": f"bytes={begin}-{segment['end']}"}
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                    while not self.stop.is_set():
                        chunk = response.read(CHUNK)
                        if not chunk:
                            break
                        offset = segment["start"] + segment["done"]
                        os.pwrite(handle_fd, chunk, offset)
                        with self.lock:
                            segment["done"] += len(chunk)
                if segment["done"] >= length:
                    return
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                if attempt == self.attempts - 1:
                    print(f"segment {index} gave up: {exc}", file=sys.stderr)
                    return
                time.sleep(min(2**attempt, 30))

    def run(self) -> int:
        self._allocate()
        started, start_bytes = time.time(), self.downloaded
        print(
            f"{self.total:,} bytes ({self.total / 2**30:.2f} GiB) over "
            f"{self.connections} connection(s); {start_bytes / 2**30:.2f} GiB already present",
            flush=True,
        )

        fd = os.open(self.target, os.O_RDWR)
        threads = [
            threading.Thread(target=self._worker, args=(i, fd), daemon=True)
            for i in range(self.connections)
        ]
        try:
            for thread in threads:
                thread.start()
            while any(t.is_alive() for t in threads):
                time.sleep(15)
                self._save()
                done = self.downloaded
                elapsed = max(time.time() - started, 1e-9)
                rate = (done - start_bytes) / elapsed
                remaining = (self.total - done) / rate if rate > 0 else float("inf")
                print(
                    f"{done / 2**30:7.2f}/{self.total / 2**30:.2f} GiB "
                    f"({100 * done / self.total:5.1f}%)  {rate / 2**20:5.1f} MiB/s  "
                    f"eta {remaining / 60:5.1f} min",
                    flush=True,
                )
        except KeyboardInterrupt:
            self.stop.set()
            print("interrupted; progress recorded", file=sys.stderr)
        finally:
            for thread in threads:
                thread.join(timeout=30)
            os.close(fd)
            self._save()

        done = self.downloaded
        actual = self.target.stat().st_size
        if done < self.total or actual != self.total:
            print(
                f"incomplete: {done:,} of {self.total:,} bytes. Re-run to resume.",
                file=sys.stderr,
            )
            return 1
        self.state_path.unlink(missing_ok=True)
        print(f"complete: {actual:,} bytes", flush=True)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("-n", "--connections", type=int, default=8)
    args = parser.parse_args()
    args.target.parent.mkdir(parents=True, exist_ok=True)
    return Segmented(args.url, args.target, connections=args.connections).run()


if __name__ == "__main__":
    raise SystemExit(main())
