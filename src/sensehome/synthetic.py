"""A synthetic catalogue, its media and user interactions. No download, no real person, no real product.

Each item gets a hidden style and colour. The photo, the tags, the text and the tap sound all come
from the SAME item attributes, so the modalities agree (unlike the prototype, which paired unrelated
files). Each user has a hidden taste and a hidden "sensory weight": some users choose by look, some
by feel. The media references are self-describing strings such as
``synthetic:color=sage;texture=woven;seed=12``, so the media render again from the catalogue alone.
"""

from __future__ import annotations

import numpy as np

from .schemas import Dataset, Interaction, Item, MATERIALS, TEXTURES

STYLES = ("scandinavian", "industrial", "bohemian", "classic", "modern")
COLORS: dict[str, tuple[int, int, int]] = {
    "white": (235, 235, 230), "cream": (240, 225, 190), "light-wood": (205, 170, 120),
    "walnut-brown": (110, 70, 40), "charcoal": (55, 55, 60), "black": (25, 25, 25),
    "terracotta": (190, 95, 60), "mustard": (210, 165, 40), "burgundy": (120, 30, 45),
    "sage": (150, 170, 140), "navy": (35, 45, 90), "grey": (140, 140, 145),
}
STYLE_COLORS = {
    "scandinavian": ("white", "light-wood", "sage", "cream"),
    "industrial": ("charcoal", "black", "grey", "walnut-brown"),
    "bohemian": ("terracotta", "mustard", "cream", "sage"),
    "classic": ("burgundy", "walnut-brown", "navy", "cream"),
    "modern": ("grey", "navy", "white", "black"),
}
STYLE_MATERIALS = {
    "scandinavian": ("oak", "pine", "linen", "wool"),
    "industrial": ("steel", "leather", "walnut", "brass"),
    "bohemian": ("rattan", "wool", "linen", "velvet"),
    "classic": ("walnut", "velvet", "leather", "marble"),
    "modern": ("steel", "marble", "leather", "linen"),
}
MATERIAL_TEXTURES = {
    "oak": ("grainy", "smooth"), "walnut": ("grainy", "smooth"), "pine": ("grainy", "rough"),
    "steel": ("brushed", "smooth"), "brass": ("brushed", "smooth"), "velvet": ("plush",),
    "linen": ("woven", "rough"), "leather": ("smooth", "grainy"), "rattan": ("woven",),
    "marble": ("smooth",), "wool": ("woven", "plush", "ribbed"),
}
STYLE_WORDS = {
    "scandinavian": ("airy", "minimal", "Nordic", "bright"),
    "industrial": ("loft", "raw", "urban", "sturdy"),
    "bohemian": ("eclectic", "relaxed", "artisan", "warm"),
    "classic": ("timeless", "elegant", "traditional", "refined"),
    "modern": ("sleek", "clean-lined", "contemporary", "bold"),
}
FILLER = ("comfortable", "durable", "easy to clean", "for small rooms", "for family homes", "handmade",
          "space-saving", "with soft edges", "a favourite piece", "built to last")
# Tap-sound physics stand-in: base frequency (Hz) and decay rate (1/s) of each material.
MATERIAL_SOUND = {
    "oak": (420, 30), "walnut": (360, 30), "pine": (520, 35), "steel": (2600, 6), "brass": (1900, 5),
    "marble": (3100, 12), "leather": (210, 60), "velvet": (150, 90), "linen": (180, 80),
    "rattan": (720, 40), "wool": (120, 100),
}
SOUND_RATE = 8000
IMAGE_SIZE = 24


def _parse(ref: str) -> dict[str, str]:
    if not ref.startswith("synthetic:"):
        raise ValueError(f"not a synthetic reference: {ref!r}")
    return dict(part.split("=", 1) for part in ref[len("synthetic:"):].split(";"))


def render_image(ref: str) -> np.ndarray:
    """A 24x24 RGB uint8 'product photo': the item colour with the pattern of its texture."""
    p = _parse(ref)
    rng = np.random.default_rng(int(p["seed"]))
    n = IMAGE_SIZE
    base = np.array(COLORS[p["color"]], dtype=np.float64)
    y, x = np.mgrid[0:n, 0:n]
    t = p["texture"]
    if t == "woven":
        pattern = np.where(((x // 4) + (y // 4)) % 2 == 0, 30.0, -30.0)
    elif t == "ribbed":
        pattern = np.where((y // 3) % 2 == 0, 35.0, -35.0)
    elif t == "brushed":
        pattern = np.repeat(rng.normal(0, 22, (1, n)), n, axis=0)
    elif t == "grainy":
        pattern = rng.normal(0, 25, (n, n))
    elif t == "plush":
        pattern = np.kron(rng.normal(0, 25, (n // 8, n // 8)), np.ones((8, 8)))
    elif t == "rough":
        pattern = np.kron(rng.normal(0, 40, (n // 4, n // 4)), np.ones((4, 4)))
    else:  # smooth
        pattern = rng.normal(0, 3, (n, n))
    light = rng.uniform(-15, 15) * (x / n - 0.5)
    img = base[None, None, :] + (pattern + light)[..., None] + rng.normal(0, 4, (n, n, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def render_sound(ref: str) -> np.ndarray:
    """A 0.25 s tap on the surface: decaying partials of the material plus a little noise."""
    p = _parse(ref)
    rng = np.random.default_rng(int(p["seed"]))
    f0, decay = MATERIAL_SOUND[p["material"]]
    t = np.arange(int(0.25 * SOUND_RATE)) / SOUND_RATE
    f0 = f0 * rng.uniform(0.95, 1.05)
    wave = sum(a * np.sin(2 * np.pi * f0 * k * t) for k, a in ((1, 1.0), (2, 0.5), (3, 0.25)) if f0 * k < SOUND_RATE / 2)
    wave = wave * np.exp(-decay * t) + rng.normal(0, 0.02, t.size)
    return (wave / (np.abs(wave).max() + 1e-9)).astype(np.float32)


def make_items(n_items: int = 240, sound_share: float = 0.6, seed: int = 7) -> list[Item]:
    rng = np.random.default_rng(seed)
    cats = ("sofa", "armchair", "table", "chair", "lamp", "rug", "shelf", "bed")
    items = []
    for k in range(n_items):
        style = STYLES[k % len(STYLES)] if k < len(STYLES) else rng.choice(STYLES)
        color = rng.choice(STYLE_COLORS[style]) if rng.random() < 0.8 else rng.choice(list(COLORS))
        material = rng.choice(STYLE_MATERIALS[style]) if rng.random() < 0.8 else rng.choice(MATERIALS)
        texture = rng.choice(MATERIAL_TEXTURES[material])
        cat = rng.choice(cats)
        words = [rng.choice(STYLE_WORDS[style])] if rng.random() < 0.6 else []
        words += [color.replace("-", " ")] if rng.random() < 0.7 else []
        desc = " ".join(words + [f"{material} {cat} with a {texture} finish,", rng.choice(FILLER)])
        title = f"{str(rng.choice(STYLE_WORDS[style])).capitalize()} {material} {cat}"
        s = int(rng.integers(0, 10**6))
        items.append(
            Item(
                item_id=f"i{k:04d}",
                category=cat,
                title=title,
                description=desc,
                materials=(str(material),),
                textures=(str(texture),),
                image=f"synthetic:color={color};texture={texture};style={style};seed={s}",
                sound=f"synthetic:material={material};seed={s}" if rng.random() < sound_share else None,
                licence="CC0-1.0 (synthetic)",
            )
        )
    return items


def hidden_attributes(item: Item) -> dict[str, str]:
    p = _parse(item.image or "")
    return {"style": p["style"], "color": p["color"], "texture": item.textures[0], "material": item.materials[0]}


def make_interactions(items: list[Item], n_users: int = 300, mean_events: int = 20, temperature: float = 0.15,
                      seed: int = 7) -> list[Interaction]:
    rng = np.random.default_rng(seed + 1)
    attrs = [hidden_attributes(i) for i in items]
    style_idx = np.array([STYLES.index(a["style"]) for a in attrs])
    color_idx = np.array([list(COLORS).index(a["color"]) for a in attrs])
    tex_idx = np.array([TEXTURES.index(a["texture"]) for a in attrs])
    mat_idx = np.array([MATERIALS.index(a["material"]) for a in attrs])
    quality = rng.normal(0, 0.3, len(items))
    out: list[Interaction] = []
    for u in range(n_users):
        def pref(k):
            v = rng.dirichlet(np.full(k, 0.4))
            return v / v.max()

        sp, cp, tp, mp = pref(len(STYLES)), pref(len(COLORS)), pref(len(TEXTURES)), pref(len(MATERIALS))
        w_look, w_feel = rng.dirichlet([1.0, 1.0])
        util = w_look * (sp[style_idx] + cp[color_idx]) / 2 + w_feel * (tp[tex_idx] + mp[mat_idx]) / 2 + quality * 0.2
        logits = util / temperature + rng.gumbel(size=len(items))
        n = int(np.clip(rng.poisson(mean_events), 6, len(items) // 2))
        chosen = np.argsort(-logits)[:n]
        rng.shuffle(chosen)
        t = int(rng.integers(0, 10**6))
        for j in chosen:
            t += int(rng.integers(60, 86400))
            ev = rng.choice(["view", "save", "purchase"], p=[0.7, 0.2, 0.1])
            out.append(Interaction(user_id=f"u{u:04d}", item_id=items[j].item_id, timestamp=t, event=str(ev)))
    return out


def make_dataset(n_items: int = 240, n_users: int = 300, mean_events: int = 20, seed: int = 7) -> Dataset:
    items = make_items(n_items, seed=seed)
    return Dataset(items, make_interactions(items, n_users, mean_events, seed=seed))
