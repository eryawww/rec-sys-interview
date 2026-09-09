import pytest

from recsys.algorithms.base import REGISTRY, Rec, Recommender, build, register
from recsys.data import Dataset, Item, LoadReport, User


@pytest.fixture
def ds() -> Dataset:
    return Dataset(
        users={"u1": User("u1", "Ada", 30, "F", "Jakarta")},
        items={"i1": Item("i1", "Show One", "series", "drama")},
        events=[],
        report=LoadReport(),
    )


def test_rec_is_immutable():
    rec = Rec("i1", "Show One", "series", "drama", 1.0, "because")
    with pytest.raises(AttributeError):
        rec.score = 2.0  # type: ignore[misc]


def test_rec_helper_fills_item_fields(ds):
    class Stub(Recommender):
        name = "stub_helper"

        def recommend_popular(self, k=10):
            return [self._rec("i1", 1.0, "because")]

        def recommend_for_user(self, user_id, k=10):
            return self.recommend_popular(k)

    rec = Stub(ds).recommend_popular()[0]
    assert rec == Rec("i1", "Show One", "series", "drama", 1.0, "because")


def test_registry_round_trip(ds):
    @register
    class Registered(Recommender):
        name = "stub_registered"

        def recommend_popular(self, k=10):
            return []

        def recommend_for_user(self, user_id, k=10):
            return []

    assert REGISTRY["stub_registered"] is Registered
    assert isinstance(build("stub_registered", ds), Registered)
    del REGISTRY["stub_registered"]


def test_build_unknown_name_lists_valid_options(ds):
    import recsys.algorithms  # noqa: F401  - ensures shipped algorithms are registered

    with pytest.raises(ValueError) as exc:
        build("does_not_exist", ds)
    message = str(exc.value)
    assert "does_not_exist" in message
    assert "item_knn" in message


def test_abstract_methods_are_enforced(ds):
    class Incomplete(Recommender):
        name = "stub_incomplete"

    with pytest.raises(TypeError):
        Incomplete(ds)  # type: ignore[abstract]
