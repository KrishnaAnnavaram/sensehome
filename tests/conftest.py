import os

import pytest

from sensehome.evaluate import prepare
from sensehome.synthetic import make_dataset


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for key in list(os.environ):
        if key.startswith("SENSEHOME_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def ds():
    return make_dataset(seed=7)


@pytest.fixture(scope="session")
def small_ds():
    return make_dataset(n_items=60, n_users=40, mean_events=10, seed=3)


@pytest.fixture(scope="session")
def prep(ds):
    return prepare(ds)
