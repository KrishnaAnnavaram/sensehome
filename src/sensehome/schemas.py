"""The item catalogue and the interaction log, with validation.

Every modality of an item hangs off ONE ``item_id``: its photo, its tags, its description and its
optional tap sound. The prototype paired the i-th file of three unrelated folders. Here an item
cannot exist without its own id, and an interaction must name an item of the catalogue.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MATERIALS = ("oak", "walnut", "pine", "steel", "brass", "velvet", "linen", "leather", "rattan", "marble", "wool")
TEXTURES = ("smooth", "grainy", "woven", "ribbed", "brushed", "plush", "rough")
CATEGORIES = ("sofa", "armchair", "table", "chair", "lamp", "rug", "shelf", "bed")
EVENTS = ("view", "save", "purchase")


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    item_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
    category: Literal["sofa", "armchair", "table", "chair", "lamp", "rug", "shelf", "bed"]
    title: str = Field(min_length=2)
    description: str = ""
    materials: tuple[str, ...] = Field(min_length=1)
    textures: tuple[str, ...] = Field(min_length=1)
    image: str | None = None  # path relative to the data folder, or "synthetic:<seed>"
    sound: str | None = None  # tap sound clip: path, "synthetic:<seed>", or None (no sound for this item)
    licence: str = Field(min_length=2, description="Licence of the photo and the text")

    @field_validator("materials")
    @classmethod
    def _materials(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        bad = [m for m in v if m not in MATERIALS]
        if bad:
            raise ValueError(f"unknown materials {bad}; allowed: {list(MATERIALS)}")
        return v

    @field_validator("textures")
    @classmethod
    def _textures(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        bad = [t for t in v if t not in TEXTURES]
        if bad:
            raise ValueError(f"unknown textures {bad}; allowed: {list(TEXTURES)}")
        return v


class Interaction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    user_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    timestamp: int = Field(ge=0)
    event: Literal["view", "save", "purchase"]


@dataclass
class Dataset:
    items: list[Item]
    interactions: list[Interaction]

    def __post_init__(self) -> None:
        ids = [i.item_id for i in self.items]
        dupes = [k for k, n in Counter(ids).items() if n > 1]
        if dupes:
            raise ValueError(f"duplicate item_id values: {dupes[:5]}")
        known = set(ids)
        unknown = sorted({x.item_id for x in self.interactions if x.item_id not in known})
        if unknown:
            raise ValueError(f"interactions name items that are not in the catalogue: {unknown[:5]}")
        self.index = {iid: n for n, iid in enumerate(ids)}

    @property
    def users(self) -> list[str]:
        return sorted({x.user_id for x in self.interactions})

    def coverage_report(self) -> dict[str, float]:
        n = len(self.items)
        return {
            "items": n,
            "with_image": sum(i.image is not None for i in self.items) / n,
            "with_sound": sum(i.sound is not None for i in self.items) / n,
            "with_description": sum(bool(i.description.strip()) for i in self.items) / n,
            "users": len(self.users),
            "interactions": len(self.interactions),
        }


def load_catalogue(path: str | Path) -> list[Item]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Item.model_validate(d) for d in data["items"]]


def load_interactions(path: str | Path) -> list[Interaction]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = {"user_id", "item_id", "timestamp", "event"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"interactions CSV misses columns {sorted(missing)}")
        return [Interaction(user_id=r["user_id"], item_id=r["item_id"], timestamp=int(r["timestamp"]), event=r["event"])
                for r in reader]


def load_dataset(folder: str | Path) -> Dataset:
    d = Path(folder)
    return Dataset(load_catalogue(d / "catalogue.json"), load_interactions(d / "interactions.csv"))


def save_dataset(ds: Dataset, folder: str | Path) -> Path:
    d = Path(folder)
    d.mkdir(parents=True, exist_ok=True)
    (d / "catalogue.json").write_text(
        json.dumps({"items": [i.model_dump() for i in ds.items]}, indent=1), encoding="utf-8"
    )
    with (d / "interactions.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["user_id", "item_id", "timestamp", "event"])
        for x in ds.interactions:
            w.writerow([x.user_id, x.item_id, x.timestamp, x.event])
    return d
