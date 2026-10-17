"""Deterministic stratified k-fold assignment."""
from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable


def make_folds(items: Iterable[tuple[str, str]], k: int = 5, seed: int = 0) -> dict[str, int]:
    """Assign each (item_id, stratum) to a fold in [0, k).

    Within each stratum items are shuffled and dealt round-robin, so per-stratum
    fold counts differ by at most one. The dealing offset carries over between
    strata so overall fold sizes stay balanced too. Input order does not matter.
    """
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for item_id, stratum in items:
        by_stratum[stratum].append(item_id)

    rng = random.Random(seed)
    folds: dict[str, int] = {}
    offset = 0
    for stratum in sorted(by_stratum):
        ids = sorted(by_stratum[stratum])
        rng.shuffle(ids)
        for i, item_id in enumerate(ids):
            folds[item_id] = (offset + i) % k
        offset = (offset + len(ids)) % k
    return folds
