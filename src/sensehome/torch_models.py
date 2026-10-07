"""A learned two-tower recommender with attention fusion over the modality embeddings (extra ``torch``).

Item tower: each modality vector goes through its own linear projection. An additive attention layer
(``v . tanh(W h_m)``) gives a weight to each PRESENT modality of the item, and the weighted sum plus an
item id embedding is the item vector. User tower: a user id embedding. Loss: BPR with sampled
negatives. torch is imported inside the methods, so the core package works without it.
"""

from __future__ import annotations

import numpy as np

from .encoders import ModalityMatrix
from .models import Recommender, TrainData


class TwoTowerAttention(Recommender):
    name = "two-tower+attention"

    def __init__(self, dim: int = 32, epochs: int = 40, lr: float = 1e-2, batch: int = 256, weight_decay: float = 1e-5,
                 seed: int = 0):
        self.dim, self.epochs, self.lr, self.batch, self.wd, self.seed = dim, epochs, lr, batch, weight_decay, seed

    def fit(self, data: TrainData, feats: dict[str, ModalityMatrix]) -> "TwoTowerAttention":
        try:
            import torch
            from torch import nn
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('the two-tower model needs: pip install -e ".[torch]"') from exc

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self.modalities = sorted(feats)
        X = [torch.tensor(feats[m].X, dtype=torch.float32) for m in self.modalities]
        M = torch.tensor(np.stack([feats[m].mask for m in self.modalities], axis=1))  # (items, M)
        n_users, n_items = len(data.pos), X[0].shape[0]
        dim = self.dim

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.ModuleList([nn.Linear(x.shape[1], dim) for x in X])
                self.att_w = nn.Linear(dim, dim)
                self.att_v = nn.Parameter(torch.randn(dim) * 0.1)
                self.item_id = nn.Embedding(n_items, dim)
                self.user = nn.Embedding(n_users, dim)
                nn.init.normal_(self.item_id.weight, std=0.01)
                nn.init.normal_(self.user.weight, std=0.1)

            def items(self):
                H = torch.stack([p(x) for p, x in zip(self.proj, X)], dim=1)  # (items, M, dim)
                e = torch.tanh(self.att_w(H)) @ self.att_v  # (items, M)
                e = e.masked_fill(~M, float("-inf"))
                a = torch.softmax(e, dim=1)
                return (a.unsqueeze(-1) * H).sum(dim=1) + self.item_id.weight, a

        net = Net()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.wd)
        pairs = np.array([(u, i) for u, p in enumerate(data.pos) for i in p])
        seen = [set(p.tolist()) for p in data.pos]
        for _ in range(self.epochs):
            rng.shuffle(pairs)
            for s in range(0, len(pairs), self.batch):
                bt = pairs[s : s + self.batch]
                j = rng.integers(0, n_items, len(bt))
                for k in range(len(j)):
                    if j[k] in seen[bt[k, 0]]:
                        j[k] = rng.integers(0, n_items)
                u = torch.tensor(bt[:, 0])
                i = torch.tensor(bt[:, 1])
                jj = torch.tensor(j)
                Q, _ = net.items()
                pu = net.user(u)
                x = (pu * (Q[i] - Q[jj])).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(x).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            Q, A = net.items()
            self.Q = Q.numpy().astype(np.float64)
            self.attention = A.numpy()
            self.U = net.user.weight.numpy().astype(np.float64)
        return self

    def scores(self, user: int) -> np.ndarray:
        return self.Q @ self.U[user]
