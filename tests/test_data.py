import json

import numpy as np
import pytest
from pydantic import ValidationError

from sensehome.schemas import Dataset, Interaction, Item, load_dataset, load_interactions, save_dataset
from sensehome.split import Split, temporal_user_split
from sensehome.synthetic import (
    COLORS,
    MATERIAL_SOUND,
    SOUND_RATE,
    STYLE_COLORS,
    hidden_attributes,
    make_items,
    render_image,
    render_sound,
)

ITEM = dict(item_id="a1", category="sofa", title="Test sofa", materials=("linen",), textures=("woven",), licence="CC0-1.0")


def test_item_validation():
    Item(**ITEM)
    with pytest.raises(ValidationError):
        Item(**{**ITEM, "materials": ("plastic",)})
    with pytest.raises(ValidationError):
        Item(**{**ITEM, "category": "spaceship"})
    with pytest.raises(ValidationError):
        Item(**{**ITEM, "price": 3})


def test_dataset_rejects_duplicates_and_unknown_items():
    # Problem 2: an interaction must name an item of the catalogue. No pairing by file order.
    a = Item(**ITEM)
    with pytest.raises(ValueError, match="duplicate"):
        Dataset([a, a], [])
    with pytest.raises(ValueError, match="not in the catalogue"):
        Dataset([a], [Interaction(user_id="u", item_id="zz", timestamp=1, event="view")])


def test_save_and_load_roundtrip(tmp_path, small_ds):
    folder = save_dataset(small_ds, tmp_path / "d")
    back = load_dataset(folder)
    assert [i.item_id for i in back.items] == [i.item_id for i in small_ds.items]
    assert back.interactions == small_ds.interactions
    assert json.loads((folder / "catalogue.json").read_text())["items"][0]["licence"].startswith("CC0")


def test_interactions_csv_needs_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("user_id,item_id\nu,i\n", encoding="utf-8")
    with pytest.raises(ValueError, match="misses columns"):
        load_interactions(p)


def test_synthetic_is_deterministic(small_ds):
    from sensehome.synthetic import make_dataset

    again = make_dataset(n_items=60, n_users=40, mean_events=10, seed=3)
    assert again.interactions == small_ds.interactions
    assert np.array_equal(render_image(small_ds.items[0].image), render_image(again.items[0].image))


def test_all_modalities_come_from_the_same_item():
    # Problem 2: photo colour, tags and tap sound all follow the attributes of ONE item.
    for it in make_items(40, seed=5):
        attrs = hidden_attributes(it)
        img = render_image(it.image).reshape(-1, 3).mean(axis=0)
        assert np.abs(img - np.array(COLORS[attrs["color"]])).max() < 45
        assert attrs["material"] == it.materials[0] and attrs["texture"] == it.textures[0]
        if it.sound:
            wave = render_sound(it.sound)
            spec = np.abs(np.fft.rfft(wave))
            peak = np.fft.rfftfreq(wave.size, 1 / SOUND_RATE)[spec.argmax()]
            assert abs(peak - MATERIAL_SOUND[attrs["material"]][0]) < 0.1 * MATERIAL_SOUND[attrs["material"]][0] + 10


def test_sound_is_optional(ds):
    rep = ds.coverage_report()
    assert 0.4 < rep["with_sound"] < 0.8 and rep["with_image"] == 1.0


def test_labels_follow_preferences_not_noise(ds):
    # Problem 1: the prototype used random labels. Here a user's items share hidden attributes.
    attrs = {it.item_id: hidden_attributes(it) for it in ds.items}
    by_user = {}
    for x in ds.interactions:
        by_user.setdefault(x.user_id, []).append(attrs[x.item_id]["style"])
    top_share = np.mean([max(s.count(v) for v in set(s)) / len(s) for s in by_user.values()])
    assert top_share > 0.35  # random choice of 5 styles gives about 0.3


def test_styles_use_their_palette():
    items = make_items(100, seed=1)
    in_palette = np.mean([hidden_attributes(i)["color"] in STYLE_COLORS[hidden_attributes(i)["style"]] for i in items])
    assert in_palette > 0.7


def test_temporal_split_has_no_leakage(ds):
    split = temporal_user_split(ds.interactions, 0.2)
    last_train = {}
    for x in split.train:
        last_train[x.user_id] = max(last_train.get(x.user_id, -1), x.timestamp)
    times = {(x.user_id, x.item_id): x.timestamp for x in ds.interactions}
    for u, items in split.test.items():
        assert all(times[(u, i)] > last_train[u] for i in items)
    assert sum(len(v) for v in split.test.values()) + len(split.train) == len({(x.user_id, x.item_id) for x in ds.interactions})


def test_split_check_detects_leakage():
    x = Interaction(user_id="u", item_id="i", timestamp=1, event="view")
    with pytest.raises(AssertionError, match="leakage"):
        Split([x], {"u": {"i"}}).check()
    with pytest.raises(ValueError):
        temporal_user_split([x], 1.5)


def test_short_histories_stay_in_train():
    xs = [Interaction(user_id="u", item_id=f"i{k}", timestamp=k, event="view") for k in range(3)]
    split = temporal_user_split(xs, 0.2, min_train=3)
    assert split.test == {} and len(split.train) == 3
