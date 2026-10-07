import numpy as np
import pytest

from sensehome.encoders import FeaturePipeline, ImageStatsEncoder, ModalityMatrix, SoundBandEncoder, TagEncoder
from sensehome.evaluate import evaluate_model, run_suite
from sensehome.metrics import bootstrap_mean_ci, hit_at_k, ndcg_at_k, recall_at_k
from sensehome.models import ContentProfile, FeatureBPR, ModalityAttention, Popularity, TrainData
from sensehome.schemas import Interaction, Item
from sensehome.synthetic import render_image, render_sound


def test_image_features_see_texture_direction():
    ribbed = ImageStatsEncoder.features(render_image("synthetic:color=grey;texture=ribbed;style=modern;seed=1"))
    brushed = ImageStatsEncoder.features(render_image("synthetic:color=grey;texture=brushed;style=modern;seed=1"))
    # [64 hist, 3 mean, gx, gy, gx-gy, std, 4 bands]: ribbed changes along y, brushed along x.
    assert ribbed[69] < 0 < brushed[69]


def test_sound_features_separate_materials():
    enc = SoundBandEncoder()
    steel = enc.features(render_sound("synthetic:material=steel;seed=1"), 8000)
    wool = enc.features(render_sound("synthetic:material=wool;seed=1"), 8000)
    assert steel[16] > wool[16]  # spectral centroid
    assert steel[17] > wool[17]  # slower decay


def test_tag_encoder():
    it = Item(item_id="a", category="rug", title="Rug", materials=("wool",), textures=("woven", "plush"), licence="CC0-1.0")
    v = TagEncoder().raw(it, None)
    assert v.sum() == 3


def test_pipeline_is_fit_on_train_items_only(ds):
    train, test = ds.items[:150], ds.items[150:]
    pipe = FeaturePipeline().fit(train)
    text = next(e for e in pipe.encoders if e.name == "text")
    train_words = {w for it in train for w in text._tokens(it)}
    assert set(text.vocab) == train_words
    feats = pipe.transform(test)
    assert feats["look"].X.shape[0] == len(test)
    assert pipe.fitted_on == frozenset(i.item_id for i in train)


def test_missing_modality_has_zero_row_and_mask(ds):
    feats = FeaturePipeline().fit(ds.items).transform(ds.items)
    m = feats["sound"]
    no_sound = [k for k, it in enumerate(ds.items) if it.sound is None]
    assert not m.mask[no_sound].any() and np.allclose(m.X[no_sound], 0)
    assert np.allclose(np.linalg.norm(m.X[m.mask], axis=1), 1)


def test_transform_before_fit_raises(ds):
    with pytest.raises(RuntimeError):
        FeaturePipeline().transform(ds.items[:2])


def test_metrics():
    assert recall_at_k([1, 2, 3], {2, 9}, 2) == 0.5
    assert hit_at_k([1, 2, 3], {3}, 2) == 0.0
    assert ndcg_at_k([5, 1], {5}, 2) == pytest.approx(1.0)
    assert ndcg_at_k([1, 5], {5}, 2) == pytest.approx(1 / np.log2(3))
    assert recall_at_k([1], set(), 1) == 0.0
    lo, hi = bootstrap_mean_ci([0, 1] * 50)
    assert lo < 0.5 < hi and bootstrap_mean_ci([]) == (0.0, 0.0)


def _toy():
    """Two item groups. Modality 'good' separates them, modality 'noise' is random."""
    rng = np.random.default_rng(0)
    n = 40
    group = np.arange(n) % 2
    good = np.zeros((n, 2))
    good[np.arange(n), group] = 1
    noise = rng.normal(size=(n, 8))
    noise /= np.linalg.norm(noise, axis=1, keepdims=True)
    feats = {"good": ModalityMatrix(good, np.ones(n, bool)), "noise": ModalityMatrix(noise, np.ones(n, bool))}
    item_index = {f"i{k}": k for k in range(n)}
    xs = []
    for u in range(30):
        g = u % 2
        picks = rng.choice(np.where(group == g)[0], 8, replace=False)
        xs += [Interaction(user_id=f"u{u}", item_id=f"i{p}", timestamp=t, event="view") for t, p in enumerate(picks)]
    return TrainData.build(xs, item_index), feats, group


def test_attention_learns_to_ignore_a_noise_modality():
    # Problem 4: the attention is real and learned, not a plain concatenation.
    data, feats, _ = _toy()
    att = ModalityAttention(["good", "noise"], epochs=300).fit(data, feats)
    a = att.alpha(0)
    assert a.sum() == pytest.approx(1.0)
    assert a[0] > 0.6 > a[1]


def test_attention_masks_missing_modalities():
    data, feats, _ = _toy()
    feats["noise"] = ModalityMatrix(feats["noise"].X * 0, np.zeros(40, bool))
    att = ModalityAttention(["good", "noise"], epochs=10).fit(data, feats)
    assert att.alpha(0)[1] == 0.0


def test_content_and_bpr_rank_the_right_group():
    data, feats, group = _toy()
    for model in (ContentProfile(["good"]), FeatureBPR(dim=8, epochs=40, seed=1)):
        model.fit(data, feats)
        top = model.recommend(0, 5, exclude=data.pos[0])
        assert (group[top] == 0).mean() >= 0.8, model.name
        assert not set(top) & set(data.pos[0].tolist())


def test_popularity():
    data, feats, _ = _toy()
    pop = Popularity().fit(data, feats)
    assert pop.scores(0).sum() == sum(len(p) for p in data.pos)


def test_suite_on_synthetic_data(ds):
    # Problem 3 and 5: a real top-K task with baselines and ablations.
    rows, prep = run_suite(ds, k=10, seed=7, bpr_epochs=10)
    by = {r.model: r for r in rows}
    full = "content[look+feel+text+sound]"
    assert by[full].recall > by["popularity"].recall + 0.05
    assert by["attention[look+feel+text+sound]"].recall > by["popularity"].recall + 0.05
    assert {"content[look]", "content[feel]", "content[text]", "content[sound]", "content[look+feel+text]"} <= set(by)
    assert all(0 <= r.recall <= 1 and r.recall_ci[0] <= r.recall <= r.recall_ci[1] for r in rows)
    assert by["popularity"].coverage < 0.2 < by[full].coverage


def test_torch_two_tower(small_ds):
    pytest.importorskip("torch")
    from sensehome.evaluate import prepare
    from sensehome.torch_models import TwoTowerAttention

    prep = prepare(small_ds)
    model = TwoTowerAttention(epochs=3, seed=0).fit(prep.data, prep.feats)
    assert model.scores(0).shape == (len(small_ds.items),)
    sound = model.modalities.index("sound")
    no_sound = ~prep.feats["sound"].mask
    assert np.allclose(model.attention[no_sound, sound], 0)
    assert np.allclose(model.attention.sum(axis=1), 1, atol=1e-5)
    assert 0 <= evaluate_model(model, prep).recall <= 1
