"""The swappable seam.

The interface is exactly the two functions Task.pdf 2.1.2 and 2.1.3
specify. Nothing wraps them and nothing decomposes them: an algorithm is
one file implementing both, registered by name and selected with the
RECSYS_ALGO environment variable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from recsys.data import Dataset


@dataclass(frozen=True, slots=True)
class Rec:
    """One recommended item.

    `reason` is the algorithm's own explanation, computed while ranking.
    It is always populated, which is what lets the LLM layer be a pure
    upgrade rather than a dependency.
    """

    item_id: str
    title: str
    content_type: str
    genre: str
    score: float
    reason: str


class Recommender(ABC):
    name: ClassVar[str]

    def __init__(self, ds: Dataset) -> None:
        self.ds = ds

    @abstractmethod
    def recommend_popular(self, k: int = 10) -> list[Rec]:
        """Return the top-k items by global popularity, highest first.

        If k is larger than the catalogue, return every item.
        """

    @abstractmethod
    def recommend_for_user(self, user_id: str, k: int = 10) -> list[Rec]:
        """Return a ranked list of recommendations for the given user.

        Does not recommend items the user has already heavily watched.
        If the user is unknown or has no history, falls back to
        recommend_popular(k).
        """

    def _rec(self, item_id: str, score: float, reason: str) -> Rec:
        item = self.ds.items[item_id]
        return Rec(
            item_id=item.item_id,
            title=item.title,
            content_type=item.content_type,
            genre=item.genre,
            score=float(score),
            reason=reason,
        )


REGISTRY: dict[str, type[Recommender]] = {}


def register(cls: type[Recommender]) -> type[Recommender]:
    REGISTRY[cls.name] = cls
    return cls


def build(name: str, ds: Dataset) -> Recommender:
    try:
        cls = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<none registered>"
        raise ValueError(
            f"Unknown algorithm {name!r}. Registered algorithms: {known}. "
            f"Set RECSYS_ALGO to one of these."
        ) from None
    return cls(ds)
