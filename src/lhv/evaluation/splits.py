"""Split construction from the split keys frame provenance already carries.

The harness never infers a split key. It reads the keys ingest attached, which
is why provenance carries them in the first place.

Leakage is a failure, not a warning. A report from a leaking split is worse than
no report, because it looks like evidence.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from ..errors import LeakageError
from ..schema import Record, opt, req

__all__ = ["SplitKind", "SplitItem", "SplitDefinition", "build_split", "assert_no_leakage"]


class SplitKind(StrEnum):
    FRAME = "frame"
    ANIMAL = "animal"
    DAY = "day"
    SITE = "site"


@dataclass(frozen=True)
class SplitItem:
    """One evaluable unit and the keys any split could be built on."""

    item_id: str
    animal_key: str = ""
    day_key: str = ""
    site_key: str = ""

    def key_for(self, kind: SplitKind) -> str:
        if kind is SplitKind.FRAME:
            return self.item_id
        if kind is SplitKind.ANIMAL:
            return self.animal_key
        if kind is SplitKind.DAY:
            return self.day_key
        return self.site_key


@dataclass(frozen=True)
class SplitDefinition(Record):
    """A split, recorded so a report can be regenerated from it exactly."""

    SCHEMA_NAME = "split_definition"
    SCHEMA_VERSION = "1"

    kind: SplitKind = req()
    seed: int = req()
    test_fraction: float = req()
    train_keys: tuple[str, ...] = opt(())
    test_keys: tuple[str, ...] = opt(())
    train_item_ids: tuple[str, ...] = opt(())
    test_item_ids: tuple[str, ...] = opt(())

    @property
    def reference(self) -> str:
        return (
            f"{self.kind}-disjoint:seed={self.seed}:test_fraction={self.test_fraction}:"
            f"train={len(self.train_keys)}:test={len(self.test_keys)}"
        )

    def overlapping_keys(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.train_keys) & set(self.test_keys)))


def build_split(
    items: Iterable[SplitItem],
    kind: SplitKind,
    *,
    test_fraction: float,
    seed: int,
) -> SplitDefinition:
    """Partition items so that no key of the requested kind spans both sides."""
    items = list(items)
    missing = [item.item_id for item in items if not item.key_for(kind)]
    if missing:
        # A split cannot be built on a key the provenance never carried. Saying
        # so is better than silently pooling everything under the empty key.
        raise LeakageError(f"{kind} (key absent from provenance)", tuple(sorted(missing)[:5]))

    keys = sorted({item.key_for(kind) for item in items})
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)

    target = max(1, round(len(shuffled) * test_fraction)) if shuffled else 0
    target = min(target, max(len(shuffled) - 1, 0)) if len(shuffled) > 1 else target
    test_keys = set(shuffled[:target])
    train_keys = [key for key in keys if key not in test_keys]

    return SplitDefinition(
        kind=kind,
        seed=seed,
        test_fraction=test_fraction,
        train_keys=tuple(train_keys),
        test_keys=tuple(sorted(test_keys)),
        train_item_ids=tuple(sorted(i.item_id for i in items if i.key_for(kind) not in test_keys)),
        test_item_ids=tuple(sorted(i.item_id for i in items if i.key_for(kind) in test_keys)),
    )


def assert_no_leakage(split: SplitDefinition, items: Iterable[SplitItem] | None = None) -> None:
    """Refuse a split whose disjointness key spans both partitions."""
    overlapping = split.overlapping_keys()
    if overlapping:
        raise LeakageError(str(split.kind), overlapping)

    if items is None:
        return
    by_id = {item.item_id: item for item in items}
    train = {by_id[i].key_for(split.kind) for i in split.train_item_ids if i in by_id}
    test = {by_id[i].key_for(split.kind) for i in split.test_item_ids if i in by_id}
    shared = tuple(sorted(train & test))
    if shared:
        raise LeakageError(str(split.kind), shared)
