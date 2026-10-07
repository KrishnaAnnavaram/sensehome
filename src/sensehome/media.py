"""Load the photo and the tap sound of an item, from a file or from a synthetic reference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .schemas import Item
from .synthetic import SOUND_RATE, render_image, render_sound


class MediaSource:
    """Resolve ``Item.image`` and ``Item.sound``. Files need the extra ``media`` (Pillow, soundfile)."""

    def __init__(self, root: str | Path = ".", image_size: int = 64):
        self.root = Path(root)
        self.image_size = image_size

    def image(self, item: Item) -> np.ndarray | None:
        if item.image is None:
            return None
        if item.image.startswith("synthetic:"):
            return render_image(item.image)
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - depends on the optional extra
            raise RuntimeError('photos on disk need: pip install -e ".[media]"') from exc
        with Image.open(self.root / item.image) as im:  # pragma: no cover - needs files
            return np.asarray(im.convert("RGB").resize((self.image_size, self.image_size)))

    def sound(self, item: Item) -> tuple[np.ndarray, int] | None:
        if item.sound is None:
            return None
        if item.sound.startswith("synthetic:"):
            return render_sound(item.sound), SOUND_RATE
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError('audio on disk needs: pip install -e ".[media]"') from exc
        data, rate = sf.read(self.root / item.sound, dtype="float32", always_2d=True)  # pragma: no cover
        return data.mean(axis=1), rate  # pragma: no cover
