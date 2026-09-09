"""Contract every registered algorithm must satisfy.

Parametrized over REGISTRY, so adding an algorithm adds its coverage
automatically. Do not add algorithm-specific assertions here.
"""

from pathlib import Path

import pytest

import recsys.algorithms  # noqa: F401  - triggers registration
from recsys.algorithms.base import REGISTRY, Rec
from recsys.algorithms.signals import Signals
from recsys.data import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def signals(ds):
    return Signals(ds)


@pytest.fixture(params=sorted(REGISTRY), scope="module")
def algo(request, ds):
    return REGISTRY[request.param](ds)


def test_registry_is_not_empty():
    assert REGISTRY


def test_popular_returns_k_items(algo):
    assert len(algo.recommend_popular(k=5)) == 5


def test_popular_k_larger_than_catalogue_returns_all(algo, ds):
    assert len(algo.recommend_popular(k=10_000)) == len(ds.items)


def test_popular_sorted_descending(algo):
    scores = [r.score for r in algo.recommend_popular(k=20)]
    assert scores == sorted(scores, reverse=True)


def test_popular_has_no_duplicates(algo):
    recs = algo.recommend_popular(k=50)
    assert len({r.item_id for r in recs}) == len(recs)


def test_popular_is_identical_for_every_caller(algo):
    assert algo.recommend_popular(k=10) == algo.recommend_popular(k=10)


def test_for_user_respects_k(algo, ds):
    user_id = ds.user_ids[0]
    assert len(algo.recommend_for_user(user_id, k=5)) == 5


def test_for_user_sorted_descending(algo, ds):
    scores = [r.score for r in algo.recommend_for_user(ds.user_ids[0], k=20)]
    assert scores == sorted(scores, reverse=True)


def test_for_user_has_no_duplicates(algo, ds):
    recs = algo.recommend_for_user(ds.user_ids[0], k=50)
    assert len({r.item_id for r in recs}) == len(recs)


def test_for_user_excludes_heavily_watched(algo, ds, signals):
    for user_id in ds.user_ids[:25]:
        for rec in algo.recommend_for_user(user_id, k=20):
            assert not signals.heavily_watched(user_id, rec.item_id)


def test_unknown_user_falls_back_to_popular(algo):
    assert algo.recommend_for_user("u999", k=5) == algo.recommend_popular(k=5)


def test_every_rec_carries_a_reason(algo, ds):
    for rec in algo.recommend_for_user(ds.user_ids[0], k=10):
        assert isinstance(rec, Rec)
        assert rec.reason.strip()
    for rec in algo.recommend_popular(k=10):
        assert rec.reason.strip()
