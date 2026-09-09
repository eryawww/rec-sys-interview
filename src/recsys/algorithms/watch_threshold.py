"""Supervised watch-threshold classifier.

Ranks items by the predicted probability that an interaction between a
user and an item would exceed WATCH_THRESHOLD_SECONDS of watch time:

    P(watch_seconds > K | age, gender, region, content_type, genre, title)

Features come only from users.csv and items.csv. No column of the event
being predicted is an input, which is what lets the model score a pair
that has never occurred; `event_type` and `watch_seconds` appear on the
label side alone.

Two properties of this design are worth stating plainly, because they
bound what the algorithm can do:

Its only view of a user is (age, gender, region). Two users with the same
triple receive the same item ordering, so the ceiling on distinct ranked
lists is the number of distinct demographic profiles - 178 across the 220
users in `data/`, with 38 profiles shared by more than one user. It
personalises by demographic segment, not by watch history. The per-user
lists still differ where the heavily-watched filter removes different
items.

Its offline accuracy on `data/` is chance. Sweeping K from 30 to 3000
gives AUC 0.476-0.515 under a random split and 0.476-0.509 grouped by
item, against 0.522 for deliberately shuffled labels; accuracy never
beats the majority-class rate. At K=600 specifically: AUC 0.515, accuracy
0.620 against a 0.653 majority rate. `research/binary_threshold_clf.py`
reproduces the sweep. The cause is in the data rather than the model -
`watch_seconds` is drawn per `event_type` and the features carry no
association with either (see docs/recommender-research.md).
"""

from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer

from recsys.algorithms.base import Rec, Recommender, register
from recsys.algorithms.popularity import PopularityRecommender
from recsys.algorithms.signals import Signals
from recsys.data import Dataset

DEFAULT_WATCH_THRESHOLD_SECONDS = 600


def _default_threshold() -> int:
    return int(os.getenv("RECSYS_WATCH_THRESHOLD_SECONDS", str(DEFAULT_WATCH_THRESHOLD_SECONDS)))


def _one_hot(values: list[str], vocabulary: list[str]) -> np.ndarray:
    index = {value: n for n, value in enumerate(vocabulary)}
    out = np.zeros((len(values), len(vocabulary)), dtype=np.float64)
    for row, value in enumerate(values):
        column = index.get(value)
        if column is not None:
            out[row, column] = 1.0
    return out


@register
class WatchThresholdRecommender(Recommender):
    """Rank by P(watch_seconds > K) from a gradient-boosted tree."""

    name = "watch_threshold"

    def __init__(self, ds: Dataset, threshold_seconds: int | None = None) -> None:
        super().__init__(ds)
        self.signals = Signals(ds)
        self._fallback = PopularityRecommender(ds)
        self.threshold_seconds = (
            _default_threshold() if threshold_seconds is None else threshold_seconds
        )

        self._item_features = self._build_item_features()
        self._user_features = self._build_user_features()
        self._model = self._fit()

    # ---- features -------------------------------------------------

    def _build_item_features(self) -> np.ndarray:
        """One row per item id, in Dataset.item_ids order."""
        items = [self.ds.items[i] for i in self.signals.item_ids]
        titles = [item.title for item in items]

        # Unigrams over titles. Fit on the catalogue rather than on the
        # event rows so the vocabulary does not depend on how often an
        # item happens to be watched.
        self._vectorizer = CountVectorizer(lowercase=True)
        title_counts = self._vectorizer.fit_transform(titles).toarray().astype(np.float64)

        self._content_types = sorted({item.content_type for item in items})
        self._genres = sorted({item.genre for item in items})
        return np.hstack(
            [
                _one_hot([item.content_type for item in items], self._content_types),
                _one_hot([item.genre for item in items], self._genres),
                title_counts,
            ]
        )

    def _build_user_features(self) -> np.ndarray:
        """One row per user id, in Dataset.user_ids order."""
        users = [self.ds.users[u] for u in self.signals.user_ids]
        self._genders = sorted({user.gender for user in users})
        self._regions = sorted({user.region for user in users})
        # Age stays numeric: the tree splits it directly, and one-hot
        # encoding 41 distinct ages would spend a feature per value.
        ages = np.array([[float(user.age)] for user in users])
        return np.hstack(
            [
                ages,
                _one_hot([user.gender for user in users], self._genders),
                _one_hot([user.region for user in users], self._regions),
            ]
        )

    def _rows_for(self, user_rows: np.ndarray, item_rows: np.ndarray) -> np.ndarray:
        return np.hstack([self._user_features[user_rows], self._item_features[item_rows]])

    # ---- training -------------------------------------------------

    def _fit(self) -> HistGradientBoostingClassifier | None:
        """Fit on one row per event. Returns None if the label is constant.

        A dataset where every event falls on the same side of the
        threshold carries no decision boundary to learn. Returning None
        makes `recommend_for_user` degrade to popularity rather than
        raising at request time.
        """
        events = self.ds.events
        if not events:
            return None

        user_rows = np.fromiter(
            (self.signals.user_index[e.user_id] for e in events), dtype=np.intp, count=len(events)
        )
        item_rows = np.fromiter(
            (self.signals.item_index[e.item_id] for e in events), dtype=np.intp, count=len(events)
        )
        labels = np.fromiter(
            (e.watch_seconds > self.threshold_seconds for e in events),
            dtype=np.int8,
            count=len(events),
        )
        if len(np.unique(labels)) < 2:
            return None

        model = HistGradientBoostingClassifier(random_state=0)
        model.fit(self._rows_for(user_rows, item_rows), labels)
        return model

    # ---- interface ------------------------------------------------

    def recommend_popular(self, k: int = 10) -> list[Rec]:
        return self._fallback.recommend_popular(k)

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        if self._model is None or user_id not in self.signals.user_index:
            return self.recommend_popular(k)
        if not self.signals.affinity[self.signals.user_index[user_id]].any():
            # The model could score this user from demographics alone,
            # but the Recommender contract specifies popularity for a
            # user with no history and the registry stays uniform.
            return self.recommend_popular(k)

        n_items = len(self.signals.item_ids)
        user_rows = np.full(n_items, self.signals.user_index[user_id], dtype=np.intp)
        item_rows = np.arange(n_items, dtype=np.intp)
        scores = self._model.predict_proba(self._rows_for(user_rows, item_rows))[:, 1]

        scores[self.signals.heavy_mask(user_id)] = -np.inf
        order = np.argsort(-scores)

        minutes = self.threshold_seconds // 60
        recs: list[Rec] = []
        for idx in order:
            if len(recs) == k:
                break
            score = scores[idx]
            if not np.isfinite(score):
                break
            item = self.ds.items[self.signals.item_ids[idx]]
            recs.append(
                self._rec(
                    item.item_id,
                    score,
                    f"{score:.0%} predicted chance you watch this {item.genre} "
                    f"{item.content_type} past {minutes} minutes",
                )
            )

        if len(recs) < k:
            recs.extend(self._top_up(user_id, k - len(recs), {r.item_id for r in recs}))
        return recs

    def _top_up(self, user_id: str, needed: int, already: set[str]) -> list[Rec]:
        """Fill the tail with popular items when candidates run out.

        Every item carries a finite probability, so this only engages
        when k exceeds the catalogue minus the user's heavily-watched
        items. Returning fewer than k would violate the interface.
        """
        mask = self.signals.heavy_mask(user_id)
        filler: list[Rec] = []
        for rec in self._fallback.recommend_popular(len(self.signals.item_ids)):
            if len(filler) == needed:
                break
            idx = self.signals.item_index[rec.item_id]
            if mask[idx] or rec.item_id in already:
                continue
            filler.append(rec)
        return filler
