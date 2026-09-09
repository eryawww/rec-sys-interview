"""Global popularity: the same ranking for everyone.

Also the fallback every other algorithm uses for an unknown user, and the
baseline the personalized algorithms are read against.
"""

from __future__ import annotations

import numpy as np

from recsys.algorithms.base import Rec, Recommender, register
from recsys.algorithms.signals import Signals
from recsys.data import Dataset


@register
class PopularityRecommender(Recommender):
    name = "popularity"

    def __init__(self, ds: Dataset) -> None:
        super().__init__(ds)
        self.signals = Signals(ds)
        # Stable ordering: break score ties by item_id so the ranking is
        # reproducible across runs.
        self._order = np.lexsort(
            (np.array(self.signals.item_ids), -self.signals.popularity)
        )
        self._rank_of = {
            self.signals.item_ids[idx]: n + 1 for n, idx in enumerate(self._order)
        }

    def _reason(self, item_id: str) -> str:
        return f"Popular right now - #{self._rank_of[item_id]} across all viewers"

    def recommend_popular(self, k: int = 10) -> list[Rec]:
        return [
            self._rec(
                self.signals.item_ids[idx],
                self.signals.popularity[idx],
                self._reason(self.signals.item_ids[idx]),
            )
            for idx in self._order[:k]
        ]

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        if user_id not in self.ds.users:
            return self.recommend_popular(k)
        mask = self.signals.heavy_mask(user_id)
        recs: list[Rec] = []
        for idx in self._order:
            if mask[idx]:
                continue
            item_id = self.signals.item_ids[idx]
            recs.append(
                self._rec(item_id, self.signals.popularity[idx], self._reason(item_id))
            )
            if len(recs) == k:
                break
        return recs
