"""Feature engineering shared by the shipped algorithms.

These are modeling choices, not facts about the file format, which is why
they live under `algorithms/` rather than in `data.py`. An algorithm is
free to ignore this module and read `Dataset.events` directly.

The signal is a function of `event_type` alone. `watch_seconds` is not
used as a strength signal: within play/like/complete it is
indistinguishable from Uniform(120, 3600) (KS p = 0.39/0.11/0.005) and
correlates -0.009 with an item's interaction count, so weighting by it
would weight noise. Its one use here is the heavily-watched threshold,
which is a behavioural definition rather than a measure of interest.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict

import numpy as np

from recsys.data import Dataset

DEFAULT_EVENT_WEIGHTS: dict[str, float] = {
    "complete": 3.0,
    "like": 2.5,
    "save": 2.0,
    "play": 1.0,
    "pause": 0.0,
    "skip": -1.0,
}


def _default_heavy_watch_seconds() -> int:
    return int(os.getenv("RECSYS_HEAVY_WATCH_SECONDS", "1800"))


class Signals:
    """Affinity and popularity arrays derived from a Dataset."""

    def __init__(
        self,
        ds: Dataset,
        weights: dict[str, float] | None = None,
        heavy_watch_seconds: int | None = None,
    ) -> None:
        self.ds = ds
        self.weights = dict(weights or DEFAULT_EVENT_WEIGHTS)
        self.heavy_watch_seconds = (
            _default_heavy_watch_seconds() if heavy_watch_seconds is None else heavy_watch_seconds
        )

        self.user_ids = ds.user_ids
        self.item_ids = ds.item_ids
        self.user_index = {u: n for n, u in enumerate(self.user_ids)}
        self.item_index = {i: n for n, i in enumerate(self.item_ids)}

        n_users, n_items = len(self.user_ids), len(self.item_ids)
        self.affinity = np.zeros((n_users, n_items), dtype=np.float64)

        seconds: defaultdict[tuple[str, str], int] = defaultdict(int)
        completed: set[tuple[str, str]] = set()
        history: defaultdict[str, list[str]] = defaultdict(list)

        for event in ds.events:
            row = self.user_index[event.user_id]
            col = self.item_index[event.item_id]
            self.affinity[row, col] += self.weights.get(event.event_type, 0.0)

            pair = (event.user_id, event.item_id)
            seconds[pair] += event.watch_seconds
            if event.event_type == "complete":
                completed.add(pair)
            history[event.user_id].append(event.item_id)

        self.popularity = self.affinity.sum(axis=0)
        self._seconds = dict(seconds)
        self._completed = completed
        # Events arrive in file order, which is chronological for this
        # fixture, so reversing puts the most recent interaction first.
        self._history = {
            user: list(dict.fromkeys(reversed(seen))) for user, seen in history.items()
        }

        self._heavy_masks: dict[str, np.ndarray] = {}

    def heavily_watched(self, user_id: str, item_id: str) -> bool:
        pair = (user_id, item_id)
        if pair in self._completed:
            return True
        return self._seconds.get(pair, 0) >= self.heavy_watch_seconds

    def heavy_mask(self, user_id: str) -> np.ndarray:
        """Boolean array over item_ids: True where the item is heavily watched."""
        cached = self._heavy_masks.get(user_id)
        if cached is not None:
            return cached
        mask = np.zeros(len(self.item_ids), dtype=bool)
        for item_id in self._history.get(user_id, ()):
            if self.heavily_watched(user_id, item_id):
                mask[self.item_index[item_id]] = True
        self._heavy_masks[user_id] = mask
        return mask

    def watched_items(self, user_id: str) -> list[str]:
        """Distinct item ids the user interacted with, most recent first."""
        return list(self._history.get(user_id, ()))

    def recent_titles(self, user_id: str, n: int = 3) -> list[str]:
        return [self.ds.items[i].title for i in self.watched_items(user_id)[:n]]

    def top_genres(self, user_id: str, n: int = 3) -> list[str]:
        counts = Counter(self.ds.items[i].genre for i in self.watched_items(user_id))
        return [genre for genre, _ in counts.most_common(n)]
