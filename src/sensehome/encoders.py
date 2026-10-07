"""Item encoders, one for each sense, and the feature pipeline that is fit on TRAIN items only.

| Modality | Offline encoder | Optional encoder |
|---|---|---|
| look  | colour histogram + texture statistics of the photo | CLIP ViT-B/32 (extra ``clip``) |
| feel  | multi-hot material and texture tags | - |
| text  | TF-IDF of title and description | - |
| sound | log band energies, centroid and decay of the tap sound | - |
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from .media import MediaSource
from .schemas import MATERIALS, TEXTURES, Item

MODALITIES = ("look", "feel", "text", "sound")


class Encoder(Protocol):
    name: str

    def fit(self, items: Sequence[Item], media: MediaSource) -> "Encoder": ...

    def raw(self, item: Item, media: MediaSource) -> np.ndarray | None: ...


class ImageStatsEncoder:
    """Colour histogram (4x4x4 bins), mean colour, gradient energy by direction, and FFT band energy."""

    name = "look"

    def fit(self, items, media):
        return self

    @staticmethod
    def features(img: np.ndarray) -> np.ndarray:
        a = np.asarray(img, dtype=np.float64)
        q = np.clip((a // 64).astype(int), 0, 3)
        idx = q[..., 0] * 16 + q[..., 1] * 4 + q[..., 2]
        hist = np.bincount(idx.ravel(), minlength=64) / idx.size
        mean = a.reshape(-1, 3).mean(axis=0) / 255.0
        g = a @ np.array([0.299, 0.587, 0.114])
        gx, gy = np.abs(np.diff(g, axis=1)).mean() / 255, np.abs(np.diff(g, axis=0)).mean() / 255
        spec = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))
        h, w = g.shape
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
        total = spec.sum() + 1e-9
        bands = [spec[(r >= lo) & (r < hi)].sum() / total for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.5))]
        return np.concatenate([hist, mean, [gx, gy, gx - gy, g.std() / 255], bands])

    def raw(self, item, media):
        img = media.image(item)
        return None if img is None else self.features(img)


class TagEncoder:
    name = "feel"
    vocab = [f"m:{m}" for m in MATERIALS] + [f"t:{t}" for t in TEXTURES]

    def fit(self, items, media):
        return self

    def raw(self, item, media):
        v = np.zeros(len(self.vocab))
        for m in item.materials:
            v[self.vocab.index(f"m:{m}")] = 1
        for t in item.textures:
            v[self.vocab.index(f"t:{t}")] = 1
        return v


_WORD = re.compile(r"[a-z]+")


class TextTfidfEncoder:
    name = "text"

    def __init__(self, min_df: int = 1):
        self.min_df = min_df
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)

    @staticmethod
    def _tokens(item: Item) -> list[str]:
        return _WORD.findall(f"{item.title} {item.description}".lower())

    def fit(self, items, media):
        df: Counter = Counter()
        for it in items:
            df.update(set(self._tokens(it)))
        words = sorted(w for w, n in df.items() if n >= self.min_df)
        self.vocab = {w: k for k, w in enumerate(words)}
        n = len(items)
        self.idf = np.array([math.log((1 + n) / (1 + df[w])) + 1 for w in words])
        return self

    def raw(self, item, media):
        v = np.zeros(len(self.vocab))
        for w, c in Counter(self._tokens(item)).items():
            if w in self.vocab:
                v[self.vocab[w]] = c
        if not v.any():
            return None
        return v * self.idf


class SoundBandEncoder:
    """16 log-spaced band energies (dB relative to the loudest band), spectral centroid and decay."""

    name = "sound"

    def __init__(self, bands: int = 16, f_min: float = 60.0):
        self.bands, self.f_min = bands, f_min

    def fit(self, items, media):
        return self

    def features(self, wave: np.ndarray, rate: int) -> np.ndarray:
        x = np.asarray(wave, dtype=np.float64)
        mag = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
        freqs = np.fft.rfftfreq(x.size, 1 / rate)
        edges = np.geomspace(self.f_min, rate / 2, self.bands + 1)
        energy = np.array([mag[(freqs >= lo) & (freqs < hi)].sum() for lo, hi in zip(edges[:-1], edges[1:])])
        db = 10 * np.log10(energy + 1e-12)
        db = np.maximum(db - db.max(), -60) / 60
        centroid = (freqs * mag).sum() / (mag.sum() + 1e-12) / (rate / 2)
        half = x.size // 2
        decay = math.log((np.sqrt((x[half:] ** 2).mean()) + 1e-9) / (np.sqrt((x[:half] ** 2).mean()) + 1e-9))
        return np.concatenate([db, [centroid, decay / 10]])

    def raw(self, item, media):
        s = media.sound(item)
        return None if s is None else self.features(*s)


class CLIPImageEncoder:  # pragma: no cover - optional extra, downloads a model
    name = "look"

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError('CLIP needs: pip install -e ".[clip]"') from exc
        self._model = CLIPModel.from_pretrained(model_name).eval()
        self._proc = CLIPProcessor.from_pretrained(model_name)

    def fit(self, items, media):
        return self

    def raw(self, item, media):
        import torch

        img = media.image(item)
        if img is None:
            return None
        with torch.no_grad():
            emb = self._model.get_image_features(**self._proc(images=[img], return_tensors="pt"))
        return emb[0].numpy().astype(np.float64)


@dataclass
class ModalityMatrix:
    X: np.ndarray  # (n_items, dim), rows L2-normalised, zero rows where missing
    mask: np.ndarray  # (n_items,) bool, True where the item has this modality


class FeaturePipeline:
    """Fit encoders and standardisation on train items; transform any item list.

    Nothing about a test item (its text, its photo statistics) changes the fitted state.
    """

    def __init__(self, encoders: Sequence[Encoder] | None = None, media: MediaSource | None = None):
        self.encoders = list(encoders) if encoders else [ImageStatsEncoder(), TagEncoder(), TextTfidfEncoder(), SoundBandEncoder()]
        self.media = media or MediaSource()
        self.stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.fitted_on: frozenset[str] = frozenset()

    @property
    def modalities(self) -> list[str]:
        return [e.name for e in self.encoders]

    def fit(self, items: Sequence[Item]) -> "FeaturePipeline":
        for enc in self.encoders:
            enc.fit(items, self.media)
            rows = [r for r in (enc.raw(it, self.media) for it in items) if r is not None]
            if not rows:
                raise ValueError(f"no train item has the modality {enc.name!r}")
            m = np.vstack(rows)
            self.stats[enc.name] = (m.mean(axis=0), m.std(axis=0) + 1e-6)
        self.fitted_on = frozenset(it.item_id for it in items)
        return self

    def transform(self, items: Sequence[Item]) -> dict[str, ModalityMatrix]:
        if not self.stats:
            raise RuntimeError("fit the pipeline before transform")
        out = {}
        for enc in self.encoders:
            mu, sd = self.stats[enc.name]
            X = np.zeros((len(items), mu.size))
            mask = np.zeros(len(items), dtype=bool)
            for k, it in enumerate(items):
                r = enc.raw(it, self.media)
                if r is not None:
                    z = (r - mu) / sd
                    n = np.linalg.norm(z)
                    X[k] = z / n if n > 0 else z
                    mask[k] = True
            out[enc.name] = ModalityMatrix(X, mask)
        return out
