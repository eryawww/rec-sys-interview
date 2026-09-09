"""Importing this package registers every shipped algorithm."""

from recsys.algorithms import item_knn, popularity, watch_threshold  # noqa: F401
from recsys.algorithms.base import REGISTRY, Rec, Recommender, build, register

__all__ = ["REGISTRY", "Rec", "Recommender", "build", "register"]
