"""Recommenders on the fused item features (NumPy only).

- ``Popularity``: the baseline that every model must beat.
- ``ContentProfile``: a user profile for each modality; score = weighted sum of modality similarities.
  With fixed equal weights this is concatenation fusion.
- ``ModalityAttention``: the same profiles, but the weight of each modality comes from a softmax that
  depends on the user (how consistent the user's history is in that modality). The parameters are
  learned with a BPR objective on a validation item for each user.
- ``FeatureBPR``: matrix factorisation with item side features (item vector = free vector + W x
  features), trained with BPR. With ``use_features=False`` it is plain BPR-MF.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .encoders import ModalityMatrix
from .schemas import Interaction

EVENT_WEIGHT = {"view": 1.0, "save": 2.0, "purchase": 3.0}


@dataclass
class TrainData:
    user_index: dict[str, int]
    item_index: dict[str, int]
    pos: list[np.ndarray]  # per user: item positions
    wts: list[np.ndarray]  # per user: event weights

    @classmethod
    def build(cls, interactions: Sequence[Interaction], item_index: dict[str, int]) -> "TrainData":
        users = sorted({x.user_id for x in interactions})
        uidx = {u: k for k, u in enumerate(users)}
        pos: dict[int, list[int]] = defaultdict(list)
        wts: dict[int, list[float]] = defaultdict(list)
        for x in interactions:
            pos[uidx[x.user_id]].append(item_index[x.item_id])
            wts[uidx[x.user_id]].append(EVENT_WEIGHT[x.event])
        return cls(uidx, item_index, [np.array(pos[k], dtype=int) for k in range(len(users))],
                   [np.array(wts[k]) for k in range(len(users))])


class Recommender:
    name = "base"

    def fit(self, data: TrainData, feats: dict[str, ModalityMatrix]) -> "Recommender":
        raise NotImplementedError

    def scores(self, user: int) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user: int, k: int, exclude: np.ndarray | None = None) -> np.ndarray:
        s = self.scores(user).astype(np.float64).copy()
        if exclude is not None and len(exclude):
            s[exclude] = -np.inf
        order = np.lexsort((np.arange(s.size), -s))
        return order[:k]


class Popularity(Recommender):
    name = "popularity"

    def fit(self, data, feats):
        n_items = len(data.item_index)
        self.counts = np.zeros(n_items)
        for p, w in zip(data.pos, data.wts):
            np.add.at(self.counts, p, w)
        return self

    def scores(self, user):
        return self.counts


def _profiles(data: TrainData, feats: dict[str, ModalityMatrix], modalities: Sequence[str]):
    """Per user and modality: the weighted mean of the user's item vectors, and its length (consistency)."""
    n_users = len(data.pos)
    prof = {m: np.zeros((n_users, feats[m].X.shape[1])) for m in modalities}
    cons = np.zeros((n_users, len(modalities)))
    for u, (p, w) in enumerate(zip(data.pos, data.wts)):
        for j, m in enumerate(modalities):
            mm = feats[m]
            keep = mm.mask[p]
            if keep.any():
                v = (mm.X[p[keep]] * w[keep, None]).sum(axis=0) / w[keep].sum()
                cons[u, j] = np.linalg.norm(v)
                prof[m][u] = v
    return prof, cons


class ContentProfile(Recommender):
    def __init__(self, modalities: Sequence[str], weights: Sequence[float] | None = None):
        self.modalities = list(modalities)
        self.weights = np.array(weights if weights is not None else [1.0] * len(self.modalities))
        self.name = "content[" + "+".join(self.modalities) + "]"

    def fit(self, data, feats):
        self.feats = feats
        self.prof, self.cons = _profiles(data, feats, self.modalities)
        return self

    def modality_scores(self, user: int) -> np.ndarray:
        """(n_modalities, n_items) similarity of the user profile to each item, standardised per modality.

        For each modality, the similarities of the items that HAVE the modality are changed to z-scores,
        and an item without the modality gets 0 (neutral). Without this step, a modality with large
        similarity values (a short, dense vector) controls the sum, and an item with a tap sound gets a
        bonus only because it has a sound.
        """
        rows = []
        for m in self.modalities:
            mm = self.feats[m]
            s = mm.X @ self.prof[m][user]
            if mm.mask.any():
                present = s[mm.mask]
                s = np.where(mm.mask, (s - present.mean()) / (present.std() + 1e-9), 0.0)
            rows.append(s)
        return np.vstack(rows)

    def scores(self, user):
        return self.weights @ self.modality_scores(user)


class ModalityAttention(ContentProfile):
    """alpha_u = softmax(theta + beta * consistency_u); score = sum_m alpha_um * S_m(u, i)."""

    def __init__(self, modalities: Sequence[str], epochs: int = 200, lr: float = 0.5, n_neg: int = 20, seed: int = 0):
        super().__init__(modalities)
        self.name = "attention[" + "+".join(self.modalities) + "]"
        self.epochs, self.lr, self.n_neg, self.seed = epochs, lr, n_neg, seed
        self.theta = np.zeros(len(self.modalities))
        self.beta = 0.0

    def alpha(self, user: int) -> np.ndarray:
        has = self.cons[user] > 0
        z = np.where(has, self.theta + self.beta * self.cons[user], -np.inf)
        if not has.any():
            return np.full(len(self.modalities), 1 / len(self.modalities))
        e = np.exp(z - z[has].max())
        return e / e.sum()

    def scores(self, user):
        return self.alpha(user) @ self.modality_scores(user)

    def fit(self, data, feats):
        rng = np.random.default_rng(self.seed)
        n_items = len(data.item_index)
        # Inner validation: each user's LAST train item is the positive; profiles use the rest.
        inner_pos, inner_wts, val = [], [], []
        for p, w in zip(data.pos, data.wts):
            if len(p) >= 2:
                inner_pos.append(p[:-1])
                inner_wts.append(w[:-1])
                val.append(p[-1])
            else:
                inner_pos.append(p)
                inner_wts.append(w)
                val.append(-1)
        inner = TrainData(data.user_index, data.item_index, inner_pos, inner_wts)
        super().fit(inner, feats)
        users = [u for u in range(len(val)) if val[u] >= 0]
        S_pos, S_neg, C = [], [], []
        for u in users:
            S = self.modality_scores(u)
            negs = rng.choice(n_items, self.n_neg)
            negs = negs[~np.isin(negs, inner_pos[u])]
            for j in negs:
                S_pos.append(S[:, val[u]])
                S_neg.append(S[:, j])
                C.append(self.cons[u])
        if S_pos:
            S_pos, S_neg, C = np.array(S_pos), np.array(S_neg), np.array(C)
            has = C > 0
            for _ in range(self.epochs):
                z = np.where(has, self.theta + self.beta * C, -np.inf)
                z = z - z.max(axis=1, keepdims=True)
                a = np.exp(z)
                a /= a.sum(axis=1, keepdims=True)
                d = S_pos - S_neg  # (n, M)
                x = (a * d).sum(axis=1)
                g = 1 / (1 + np.exp(np.clip(x, -30, 30)))  # sigmoid(-x), the BPR gradient factor
                da = a * (d - x[:, None])  # dx/dtheta_m
                self.theta += self.lr * (g[:, None] * da).mean(axis=0)
                self.beta += self.lr * (g * (da * C).sum(axis=1)).mean()
        # Final profiles on the full train data.
        ContentProfile.fit(self, data, feats)
        return self


class FeatureBPR(Recommender):
    def __init__(self, dim: int = 32, epochs: int = 30, lr: float = 0.05, reg: float = 1e-4, batch: int = 256,
                 use_features: bool = True, seed: int = 0):
        self.dim, self.epochs, self.lr, self.reg, self.batch = dim, epochs, lr, reg, batch
        self.use_features, self.seed = use_features, seed
        self.name = "bpr+features" if use_features else "bpr"

    def fit(self, data, feats):
        rng = np.random.default_rng(self.seed)
        n_users, n_items = len(data.pos), len(data.item_index)
        F = np.hstack([feats[m].X for m in sorted(feats)]) if self.use_features else np.zeros((n_items, 1))
        self.F = F
        self.P = rng.normal(0, 0.1, (n_users, self.dim))
        self.E = rng.normal(0, 0.1, (n_items, self.dim))
        self.W = rng.normal(0, 0.01, (F.shape[1], self.dim))
        self.b = np.zeros(n_items)
        pairs = np.array([(u, i) for u, p in enumerate(data.pos) for i in p])
        user_sets = [set(p.tolist()) for p in data.pos]
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for start in range(0, len(pairs), self.batch):
                bt = pairs[start : start + self.batch]
                u, i = bt[:, 0], bt[:, 1]
                j = rng.integers(0, n_items, len(bt))
                for k in range(len(j)):  # resample negatives that the user has seen
                    tries = 0
                    while j[k] in user_sets[u[k]] and tries < 10:
                        j[k] = rng.integers(0, n_items)
                        tries += 1
                qi = self.E[i] + F[i] @ self.W
                qj = self.E[j] + F[j] @ self.W
                pu = self.P[u]
                x = (pu * (qi - qj)).sum(axis=1) + self.b[i] - self.b[j]
                g = 1 / (1 + np.exp(np.clip(x, -30, 30)))
                gp = g[:, None]
                dP = gp * (qi - qj) - self.reg * pu
                dEi = gp * pu - self.reg * self.E[i]
                dEj = -gp * pu - self.reg * self.E[j]
                dW = (F[i] - F[j]).T @ (gp * pu) / len(bt) - self.reg * self.W
                np.add.at(self.P, u, self.lr * dP)
                np.add.at(self.E, i, self.lr * dEi)
                np.add.at(self.E, j, self.lr * dEj)
                np.add.at(self.b, i, self.lr * (g - self.reg * self.b[i]))
                np.add.at(self.b, j, self.lr * (-g - self.reg * self.b[j]))
                self.W += self.lr * dW
        self.Q = self.E + F @ self.W
        return self

    def scores(self, user):
        return self.Q @ self.P[user] + self.b
