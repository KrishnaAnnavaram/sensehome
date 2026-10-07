"""The ``sensehome`` command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import Settings
from .encoders import FeaturePipeline
from .evaluate import run_suite
from .models import ContentProfile, FeatureBPR, ModalityAttention, Popularity, TrainData
from .schemas import Dataset, load_dataset, save_dataset
from .synthetic import make_dataset

MODELS = ("attention", "content", "bpr", "bpr+features", "popularity")


def _dataset(args) -> Dataset:
    folder = Path(args.data) if args.data else None
    if folder and (folder / "catalogue.json").is_file():
        return load_dataset(folder)
    if folder and args.data != Settings.from_env().data_dir:
        raise SystemExit(f"no catalogue.json in {folder}")
    return make_dataset(seed=args.seed)


def _make(name: str, modalities, seed: int):
    return {
        "attention": lambda: ModalityAttention(modalities, seed=seed),
        "content": lambda: ContentProfile(modalities),
        "bpr": lambda: FeatureBPR(use_features=False, seed=seed),
        "bpr+features": lambda: FeatureBPR(use_features=True, seed=seed),
        "popularity": lambda: Popularity(),
    }[name]()


def cmd_generate(args) -> int:
    ds = make_dataset(args.items, args.users, args.events, seed=args.seed)
    out = save_dataset(ds, args.out)
    print(f"wrote {out}/catalogue.json and {out}/interactions.csv: {ds.coverage_report()}")
    return 0


def cmd_validate(args) -> int:
    try:
        ds = _dataset(args)
    except (ValueError, OSError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(ds.coverage_report(), indent=2))
    return 0


def cmd_seeds(args) -> int:
    """Repeat the synthetic experiment for several seeds (new data and new model seeds each time)."""
    seeds = [int(x) for x in args.seeds.split(",")]
    table: dict[str, list[float]] = {}
    for seed in seeds:
        rows, _ = run_suite(make_dataset(seed=seed), k=args.k, seed=seed, ablations=False)
        for r in rows:
            table.setdefault(r.model, []).append(r.recall)
    print(f"Recall@{args.k} on synthetic data for seeds {seeds}")
    for model, vals in table.items():
        v = np.array(vals)
        print(f"{model:36s} mean {v.mean():.3f}  std {v.std(ddof=1) if v.size > 1 else 0.0:.3f}  "
              + " ".join(f"{x:.3f}" for x in v))
    return 0


def cmd_evaluate(args) -> int:
    if args.seeds:
        return cmd_seeds(args)
    ds = _dataset(args)
    extra = []
    if args.torch:
        from .torch_models import TwoTowerAttention

        extra.append(TwoTowerAttention(seed=args.seed))
    rows, prep = run_suite(ds, k=args.k, seed=args.seed, ablations=not args.no_ablations, extra=extra)
    if args.json:
        print(json.dumps([r.as_dict() for r in rows], indent=2))
        return 0
    print(f"{len(ds.items)} items, {len(ds.users)} users, {len(prep.split.train)} train interactions, "
          f"{len(prep.split.test)} test users, k={args.k}, seed={args.seed}")
    print(f"{'model':36s} {'Recall@K':>8s} {'95% CI':>15s} {'NDCG@K':>7s} {'95% CI':>15s} {'Hit@K':>6s} {'coverage':>8s}")
    for r in rows:
        print(f"{r.model:36s} {r.recall:8.3f} [{r.recall_ci[0]:.3f},{r.recall_ci[1]:.3f}] {r.ndcg:7.3f} "
              f"[{r.ndcg_ci[0]:.3f},{r.ndcg_ci[1]:.3f}] {r.hit:6.3f} {r.coverage:8.2f}")
    return 0


def _fit_all(ds: Dataset, model_name: str, seed: int):
    pipe = FeaturePipeline().fit(ds.items)
    feats = pipe.transform(ds.items)
    data = TrainData.build(sorted(ds.interactions, key=lambda x: (x.user_id, x.timestamp)), ds.index)
    model = _make(model_name, pipe.modalities, seed).fit(data, feats)
    return model, data


def cmd_train(args) -> int:
    ds = _dataset(args)
    model, data = _fit_all(ds, args.model, args.seed)
    ids = [it.item_id for it in ds.items]
    recs = {}
    for user, u in data.user_index.items():
        recs[user] = [ids[i] for i in model.recommend(u, args.k, exclude=data.pos[u])]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.name, "k": args.k, "seed": args.seed, "recommendations": recs}
    if isinstance(model, ModalityAttention):
        payload["attention"] = {"modalities": model.modalities, "theta": model.theta.tolist(), "beta": model.beta}
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"trained {model.name} on {len(ds.interactions)} interactions; wrote top-{args.k} lists for {len(recs)} users to {out}")
    return 0


def cmd_recommend(args) -> int:
    ds = _dataset(args)
    model, data = _fit_all(ds, args.model, args.seed)
    if args.user not in data.user_index:
        print(f"unknown user {args.user!r}", file=sys.stderr)
        return 1
    u = data.user_index[args.user]
    top = model.recommend(u, args.k, exclude=data.pos[u])
    print(f"user {args.user}, model {model.name}, history {len(data.pos[u])} items")
    if isinstance(model, ContentProfile):
        weights = model.alpha(u) if isinstance(model, ModalityAttention) else model.weights / model.weights.sum()
        print("modality weights: " + ", ".join(f"{m}={w:.2f}" for m, w in zip(model.modalities, weights)))
        S = model.modality_scores(u)
    for rank, i in enumerate(top, start=1):
        it = ds.items[i]
        line = f"{rank:2d}. {it.item_id} {it.title} ({', '.join(it.materials)}, {', '.join(it.textures)})"
        if isinstance(model, ContentProfile):
            line += "  " + " ".join(f"{m}:{S[j, i]:+.2f}" for j, m in enumerate(model.modalities))
        print(line)
    return 0


def build_parser() -> argparse.ArgumentParser:
    s = Settings.from_env()
    p = argparse.ArgumentParser(prog="sensehome", description="Multi-sensory furniture recommender")
    sub = p.add_subparsers(dest="command", required=True)

    def data(sp):
        sp.add_argument("--data", default=s.data_dir, help="folder with catalogue.json and interactions.csv "
                        "(default: SENSEHOME_DATA_DIR; if it is missing, a synthetic set is made in memory)")
        sp.add_argument("--seed", type=int, default=s.seed)

    sp = sub.add_parser("generate", help="write a synthetic catalogue and interaction log")
    sp.add_argument("--out", default=s.data_dir)
    sp.add_argument("--items", type=int, default=240)
    sp.add_argument("--users", type=int, default=300)
    sp.add_argument("--events", type=int, default=20, help="mean interactions for each user")
    sp.add_argument("--seed", type=int, default=s.seed)
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("validate", help="validate a catalogue and an interaction log")
    data(sp)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("evaluate", help="temporal split, baselines, fusion models and ablations")
    data(sp)
    sp.add_argument("--k", type=int, default=s.k)
    sp.add_argument("--no-ablations", action="store_true")
    sp.add_argument("--torch", action="store_true", help="also train the two-tower attention model (extra torch)")
    sp.add_argument("--seeds", help="comma list: repeat on new synthetic data for each seed and summarise")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_evaluate)

    sp = sub.add_parser("train", help="fit one model on all interactions and write top-K lists")
    data(sp)
    sp.add_argument("--model", choices=MODELS, default="attention")
    sp.add_argument("--k", type=int, default=s.k)
    sp.add_argument("--out", default=str(Path(s.model_dir) / "recommendations.json"))
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("recommend", help="show and explain the top-K items for one user")
    data(sp)
    sp.add_argument("--user", required=True)
    sp.add_argument("--model", choices=MODELS, default="attention")
    sp.add_argument("--k", type=int, default=s.k)
    sp.set_defaults(func=cmd_recommend)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    np.seterr(over="ignore")
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
