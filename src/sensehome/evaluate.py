"""The evaluation suite: baselines, single-sense models, fusion models and ablations.

All models see the same temporal split. The feature pipeline is fit on the items of the TRAIN
interactions only. Each user's train items are removed from the ranking before scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .encoders import MODALITIES, FeaturePipeline, ModalityMatrix
from .metrics import bootstrap_mean_ci, hit_at_k, ndcg_at_k, recall_at_k
from .models import ContentProfile, FeatureBPR, ModalityAttention, Popularity, Recommender, TrainData
from .schemas import Dataset
from .split import Split, temporal_user_split


@dataclass
class Row:
    model: str
    users: int
    recall: float
    recall_ci: tuple[float, float]
    ndcg: float
    ndcg_ci: tuple[float, float]
    hit: float
    coverage: float  # share of catalogue items that appear in any top-K list

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prepared:
    ds: Dataset
    split: Split
    data: TrainData
    feats: dict[str, ModalityMatrix]
    pipeline: FeaturePipeline


def prepare(ds: Dataset, test_fraction: float = 0.2, pipeline: FeaturePipeline | None = None) -> Prepared:
    split = temporal_user_split(ds.interactions, test_fraction)
    train_ids = {x.item_id for x in split.train}
    train_items = [it for it in ds.items if it.item_id in train_ids]
    pipe = (pipeline or FeaturePipeline()).fit(train_items)
    feats = pipe.transform(ds.items)
    data = TrainData.build(split.train, ds.index)
    return Prepared(ds, split, data, feats, pipe)


def evaluate_model(model: Recommender, prep: Prepared, k: int = 10, seed: int = 0) -> Row:
    recs, nds, hits = [], [], []
    shown: set[int] = set()
    for user, items in sorted(prep.split.test.items()):
        u = prep.data.user_index.get(user)
        if u is None:
            continue
        rel = {prep.ds.index[i] for i in items}
        ranked = model.recommend(u, k, exclude=prep.data.pos[u])
        shown.update(ranked.tolist())
        recs.append(recall_at_k(ranked, rel, k))
        nds.append(ndcg_at_k(ranked, rel, k))
        hits.append(hit_at_k(ranked, rel, k))
    return Row(model.name, len(recs), float(np.mean(recs)), bootstrap_mean_ci(recs, seed=seed), float(np.mean(nds)),
               bootstrap_mean_ci(nds, seed=seed), float(np.mean(hits)), len(shown) / len(prep.ds.items))


def default_models(modalities: Sequence[str] = MODALITIES, seed: int = 0, ablations: bool = True,
                   bpr_epochs: int = 30) -> list[Recommender]:
    mods = list(modalities)
    models: list[Recommender] = [Popularity()]
    if ablations:
        models += [ContentProfile([m]) for m in mods]
    models.append(ContentProfile(mods))
    if ablations and len(mods) > 1:
        models += [ContentProfile([m for m in mods if m != drop]) for drop in mods]
    models.append(ModalityAttention(mods, seed=seed))
    models += [FeatureBPR(use_features=False, seed=seed, epochs=bpr_epochs),
               FeatureBPR(use_features=True, seed=seed, epochs=bpr_epochs)]
    return models


def run_suite(ds: Dataset, k: int = 10, seed: int = 0, ablations: bool = True, extra: Sequence[Recommender] = (),
              bpr_epochs: int = 30) -> tuple[list[Row], Prepared]:
    prep = prepare(ds)
    rows = []
    for model in list(default_models(prep.pipeline.modalities, seed, ablations, bpr_epochs)) + list(extra):
        model.fit(prep.data, prep.feats)
        rows.append(evaluate_model(model, prep, k, seed))
    return rows, prep
