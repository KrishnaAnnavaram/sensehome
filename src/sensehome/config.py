"""Settings from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _get(name: str, default: str) -> str:
    v = os.environ.get("SENSEHOME_" + name, "")
    return v.strip() or default


@dataclass(frozen=True)
class Settings:
    data_dir: str = "data/synthetic"
    model_dir: str = "models"
    seed: int = 7
    k: int = 10
    image_encoder: str = "histogram"  # "histogram" (offline) or "clip"

    @classmethod
    def from_env(cls) -> "Settings":
        k = int(_get("K", "10"))
        if k < 1:
            raise ValueError("SENSEHOME_K must be 1 or more")
        enc = _get("IMAGE_ENCODER", "histogram").lower()
        if enc not in {"histogram", "clip"}:
            raise ValueError("SENSEHOME_IMAGE_ENCODER must be 'histogram' or 'clip'")
        return cls(_get("DATA_DIR", "data/synthetic"), _get("MODEL_DIR", "models"), int(_get("SEED", "7")), k, enc)
