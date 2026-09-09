from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from recsys.algorithms.signals import Signals
from recsys.algorithms.watch_threshold import (
    DEFAULT_WATCH_THRESHOLD_SECONDS,
    WatchThresholdRecommender,
)
from recsys.data import Dataset, Event, Item, LoadReport, User, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def algo(ds):
    return WatchThresholdRecommender(ds)


def test_ships_with_a_600_second_threshold(algo):
    assert DEFAULT_WATCH_THRESHOLD_SECONDS == 600
    assert algo.threshold_seconds == 600


def test_threshold_is_configurable_by_environment(ds, monkeypatch):
    monkeypatch.setenv("RECSYS_WATCH_THRESHOLD_SECONDS", "1200")
    assert WatchThresholdRecommender(ds).threshold_seconds == 1200


def test_scores_are_probabilities(ds, algo):
    scores = [r.score for r in algo.recommend_for_user(ds.user_ids[0], k=20)]
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_reason_states_the_threshold_in_minutes(ds, algo):
    reasons = [r.reason for r in algo.recommend_for_user(ds.user_ids[0], k=5)]
    assert all("past 10 minutes" in reason for reason in reasons)


def test_no_event_column_reaches_the_feature_matrix(algo, ds):
    """Features are user and item attributes only.

    The width of a feature row must equal the demographic block plus the
    item block. Any leak of event_type or watch_seconds would widen it.
    """
    expected = (
        1  # age
        + len({u.gender for u in ds.users.values()})
        + len({u.region for u in ds.users.values()})
        + len({i.content_type for i in ds.items.values()})
        + len({i.genre for i in ds.items.values()})
        + len(algo._vectorizer.vocabulary_)
    )
    rows = algo._rows_for(np.array([0]), np.array([0]))
    assert rows.shape == (1, expected)


def test_users_sharing_a_demographic_profile_are_ranked_identically(ds, algo):
    """The model sees a user only as (age, gender, region).

    Two users with the same triple therefore receive the same ordering,
    up to the heavily-watched filter. This is a documented bound on what
    the algorithm can personalise, so it is pinned rather than left to
    be rediscovered.
    """
    profiles: dict[tuple[int, str, str], list[str]] = {}
    for user_id in ds.user_ids:
        user = ds.users[user_id]
        profiles.setdefault((user.age, user.gender, user.region), []).append(user_id)
    shared = [group for group in profiles.values() if len(group) > 1]
    assert shared, "fixture is expected to contain at least one shared profile"

    signals = Signals(ds)
    first, second = shared[0][:2]
    n_items = len(signals.item_ids)

    def raw_scores(user_id: str) -> np.ndarray:
        rows = algo._rows_for(
            np.full(n_items, signals.user_index[user_id]), np.arange(n_items)
        )
        return algo._model.predict_proba(rows)[:, 1]

    assert np.allclose(raw_scores(first), raw_scores(second))


def test_constant_label_degrades_to_popularity():
    """Every event on one side of the threshold leaves nothing to learn."""
    users = {"u1": User("u1", "A", 30, "F", "Jakarta")}
    items = {
        "i1": Item("i1", "One", "movie", "drama"),
        "i2": Item("i2", "Two", "series", "romance"),
    }
    events = [
        Event("u1", "i1", "play", 10, datetime(2025, 1, 1)),
        Event("u1", "i2", "play", 20, datetime(2025, 1, 2)),
    ]
    ds = Dataset(users=users, items=items, events=events, report=LoadReport())
    algo = WatchThresholdRecommender(ds)

    assert algo._model is None
    assert algo.recommend_for_user("u1", k=2) == algo.recommend_popular(k=2)


def test_empty_event_log_does_not_raise():
    users = {"u1": User("u1", "A", 30, "F", "Jakarta")}
    items = {"i1": Item("i1", "One", "movie", "drama")}
    ds = Dataset(users=users, items=items, events=[], report=LoadReport())
    algo = WatchThresholdRecommender(ds)

    assert algo._model is None
    assert algo.recommend_for_user("u1", k=1) == algo.recommend_popular(k=1)
