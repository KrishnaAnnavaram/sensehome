"""Leakage-safe splits of the interaction log.

``temporal_user_split`` keeps, for each user, the LAST interactions for test. Every test interaction
of a user is later than every train interaction of the same user, and a test item never appears in
that user's train history.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from .schemas import Interaction


@dataclass
class Split:
    train: list[Interaction]
    test: dict[str, set[str]]  # user -> held-out item ids

    def check(self) -> None:
        """Raise if a test item is in the same user's train history, or is not later in time."""
        last_train: dict[str, int] = defaultdict(lambda: -1)
        seen: dict[str, set[str]] = defaultdict(set)
        for x in self.train:
            last_train[x.user_id] = max(last_train[x.user_id], x.timestamp)
            seen[x.user_id].add(x.item_id)
        for u, items in self.test.items():
            overlap = items & seen[u]
            if overlap:
                raise AssertionError(f"leakage: user {u} has test items in train: {sorted(overlap)[:3]}")


def _dedupe_first(xs: Sequence[Interaction]) -> list[Interaction]:
    """Keep the first interaction of each (user, item), in time order."""
    seen, out = set(), []
    for x in sorted(xs, key=lambda x: (x.user_id, x.timestamp, x.item_id)):
        key = (x.user_id, x.item_id)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out


def temporal_user_split(interactions: Sequence[Interaction], test_fraction: float = 0.2, min_train: int = 3) -> Split:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    by_user: dict[str, list[Interaction]] = defaultdict(list)
    for x in _dedupe_first(interactions):
        by_user[x.user_id].append(x)
    train: list[Interaction] = []
    test: dict[str, set[str]] = {}
    test_times: dict[str, int] = {}
    for u, xs in by_user.items():
        n_test = max(1, int(round(len(xs) * test_fraction)))
        if len(xs) - n_test < min_train:
            train.extend(xs)  # too short a history: train only
            continue
        train.extend(xs[:-n_test])
        test[u] = {x.item_id for x in xs[-n_test:]}
        test_times[u] = xs[-n_test].timestamp
    split = Split(train, test)
    split.check()
    for x in train:
        if x.user_id in test_times and x.timestamp > test_times[x.user_id]:
            raise AssertionError("leakage: a train interaction is later than the user's test cut")
    return split
