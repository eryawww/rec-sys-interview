from datetime import datetime

import numpy as np
import pytest

from recsys.algorithms.signals import DEFAULT_EVENT_WEIGHTS, Signals
from recsys.data import Dataset, Event, Item, LoadReport, User


def make_dataset(events: list[tuple[str, str, str, int]]) -> Dataset:
    users = {u: User(u, f"Name {u}", 30, "F", "Jakarta") for u in ("u1", "u2")}
    items = {i: Item(i, f"Title {i}", "series", "drama") for i in ("i1", "i2", "i3")}
    parsed = [
        Event(u, i, kind, secs, datetime(2025, 1, 1 + n, 12, 0))
        for n, (u, i, kind, secs) in enumerate(events)
    ]
    return Dataset(users=users, items=items, events=parsed, report=LoadReport())


def test_weights_come_from_event_type_only():
    ds = make_dataset([("u1", "i1", "complete", 10), ("u1", "i2", "play", 3000)])
    sig = Signals(ds)
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i1"]] == 3.0
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i2"]] == 1.0


def test_skip_lowers_score():
    ds = make_dataset([("u1", "i1", "play", 100), ("u1", "i1", "skip", 5)])
    sig = Signals(ds)
    assert sig.affinity[sig.user_index["u1"], sig.item_index["i1"]] == 0.0


def test_popularity_is_column_sum():
    ds = make_dataset([("u1", "i1", "like", 10), ("u2", "i1", "save", 10)])
    sig = Signals(ds)
    assert sig.popularity[sig.item_index["i1"]] == pytest.approx(4.5)


def test_heavily_watched_on_complete_regardless_of_seconds():
    ds = make_dataset([("u1", "i1", "complete", 1)])
    sig = Signals(ds)
    assert sig.heavily_watched("u1", "i1") is True


def test_heavily_watched_on_accumulated_seconds():
    ds = make_dataset([("u1", "i1", "play", 1000), ("u1", "i1", "play", 900)])
    sig = Signals(ds, heavy_watch_seconds=1800)
    assert sig.heavily_watched("u1", "i1") is True


def test_skipped_item_stays_recommendable():
    ds = make_dataset([("u1", "i1", "skip", 20)])
    sig = Signals(ds)
    assert sig.heavily_watched("u1", "i1") is False


def test_heavy_mask_matches_scalar_check():
    ds = make_dataset([("u1", "i1", "complete", 10), ("u1", "i2", "skip", 5)])
    sig = Signals(ds)
    mask = sig.heavy_mask("u1")
    assert mask[sig.item_index["i1"]]
    assert not mask[sig.item_index["i2"]]
    assert mask.dtype == np.bool_


def test_unknown_user_has_empty_mask_and_history():
    ds = make_dataset([("u1", "i1", "play", 10)])
    sig = Signals(ds)
    assert not sig.heavy_mask("nobody").any()
    assert sig.watched_items("nobody") == []
    assert sig.recent_titles("nobody") == []


def test_recent_titles_are_newest_first():
    ds = make_dataset(
        [("u1", "i1", "play", 10), ("u1", "i2", "play", 10), ("u1", "i3", "play", 10)]
    )
    sig = Signals(ds)
    assert sig.recent_titles("u1", n=2) == ["Title i3", "Title i2"]


def test_default_weights_are_event_type_only():
    assert DEFAULT_EVENT_WEIGHTS == {
        "complete": 3.0,
        "like": 2.5,
        "save": 2.0,
        "play": 1.0,
        "pause": 0.0,
        "skip": -1.0,
    }
