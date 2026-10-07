"""Top-K ranking metrics with bootstrap confidence intervals over users."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(list(ranked)[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    return float(bool(set(list(ranked)[:k]) & relevant))


def ndcg_at_k(ranked: Sequence[int], relevant: set[int], k: int) -> float:
    dcg = sum(1 / math.log2(r + 2) for r, item in enumerate(list(ranked)[:k]) if item in relevant)
    ideal = sum(1 / math.log2(r + 2) for r in range(min(k, len(relevant))))
    return dcg / ideal if ideal else 0.0


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, v.size, (n_boot, v.size))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
