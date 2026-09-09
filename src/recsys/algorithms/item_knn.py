"""Item-based k-nearest-neighbours over the interaction matrix.

The score for an item is a sum of contributions from items the user
already engaged with, which means the largest contributing term names the
item responsible for the recommendation. That per-item provenance is the
native `reason`, and it is the evidence the LLM layer rephrases.

Neighbours are truncated to the top TOP_NEIGHBOURS per item: the
truncated form both ranked higher and covered more of the catalogue than
the untruncated form in offline evaluation.
"""

from __future__ import annotations

import numpy as np

from recsys.algorithms.base import Rec, Recommender, register
from recsys.algorithms.popularity import PopularityRecommender
from recsys.algorithms.signals import Signals
from recsys.data import Dataset


@register
class ItemKnnRecommender(Recommender):
    name = "item_knn"
    TOP_NEIGHBOURS = 50

    def __init__(self, ds: Dataset) -> None:
        super().__init__(ds)
        self.signals = Signals(ds)
        self._fallback = PopularityRecommender(ds)

        matrix = self.signals.affinity
        norms = np.linalg.norm(matrix, axis=0)
        norms[norms == 0.0] = 1.0
        normalised = matrix / norms
        similarity = normalised.T @ normalised
        np.fill_diagonal(similarity, 0.0)

        # Keep only the strongest TOP_NEIGHBOURS per item.
        n_items = similarity.shape[0]
        keep = min(self.TOP_NEIGHBOURS, n_items - 1)
        threshold = np.partition(similarity, n_items - keep, axis=1)[:, n_items - keep][:, None]
        similarity[similarity < threshold] = 0.0
        self.similarity = similarity

    def recommend_popular(self, k: int = 10) -> list[Rec]:
        return self._fallback.recommend_popular(k)

    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        if user_id not in self.signals.user_index:
            return self.recommend_popular(k)

        row = self.signals.affinity[self.signals.user_index[user_id]]
        if not row.any():
            return self.recommend_popular(k)

        # contributions[j, i] is how much watched item j pushed up item i.
        contributions = row[:, None] * self.similarity
        scores = contributions.sum(axis=0)

        scores[self.signals.heavy_mask(user_id)] = -np.inf
        order = np.argsort(-scores)

        recs: list[Rec] = []
        for idx in order:
            if len(recs) == k:
                break
            score = scores[idx]
            if not np.isfinite(score) or score <= 0.0:
                break
            source = int(np.argmax(contributions[:, idx]))
            source_title = self.ds.items[self.signals.item_ids[source]].title
            recs.append(
                self._rec(self.signals.item_ids[idx], score, f"Because you watched {source_title}")
            )

        if len(recs) < k:
            recs.extend(self._top_up(user_id, k - len(recs), {r.item_id for r in recs}))
        return recs

    def _top_up(self, user_id: str, needed: int, already: set[str]) -> list[Rec]:
        """Fill the tail with popular items when similarity runs out.

        A user whose neighbours are all heavily watched can exhaust the
        positive-score candidates before reaching k. Returning fewer than
        k would violate the interface, so popularity completes the list.
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
