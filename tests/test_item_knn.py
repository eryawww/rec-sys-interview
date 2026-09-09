from pathlib import Path

import numpy as np
import pytest

from recsys.algorithms.item_knn import ItemKnnRecommender
from recsys.algorithms.popularity import PopularityRecommender
from recsys.algorithms.signals import Signals
from recsys.data import load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def algo(ds):
    return ItemKnnRecommender(ds)


def test_output_differs_between_users(ds, algo):
    a = [r.item_id for r in algo.recommend_for_user(ds.user_ids[0], k=10)]
    b = [r.item_id for r in algo.recommend_for_user(ds.user_ids[1], k=10)]
    assert a != b


def test_reason_names_a_watched_title(ds, algo):
    signals = Signals(ds)
    user_id = ds.user_ids[0]
    watched = {ds.items[i].title for i in signals.watched_items(user_id)}
    reasons = [r.reason for r in algo.recommend_for_user(user_id, k=5)]
    assert any(any(title in reason for title in watched) for reason in reasons)


def test_covers_far_more_of_the_catalogue_than_popularity(ds, algo):
    pop = PopularityRecommender(ds)
    knn_items = {r.item_id for u in ds.user_ids for r in algo.recommend_for_user(u, k=10)}
    pop_items = {r.item_id for u in ds.user_ids for r in pop.recommend_for_user(u, k=10)}
    assert len(knn_items) > 5 * len(pop_items)


def test_similarity_matrix_has_zero_diagonal(algo):
    assert np.allclose(np.diag(algo.similarity), 0.0)


def test_user_with_no_usable_neighbours_still_gets_k_items(ds, algo):
    recs = algo.recommend_for_user(ds.user_ids[0], k=260)
    assert len(recs) > 0
