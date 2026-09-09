from pathlib import Path

import pytest

from recsys.data import Dataset, load_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@pytest.fixture(scope="module")
def ds() -> Dataset:
    return load_dataset(DATA_DIR)


def test_loads_expected_row_counts(ds):
    assert len(ds.users) == 220
    assert len(ds.items) == 260
    assert len(ds.events) == 5000


def test_referential_integrity(ds):
    for event in ds.events:
        assert event.user_id in ds.users
        assert event.item_id in ds.items


def test_types_are_coerced(ds):
    event = ds.events[0]
    assert isinstance(event.watch_seconds, int)
    assert event.timestamp.year == 2025
    assert isinstance(next(iter(ds.users.values())).age, int)


def test_ids_are_sorted_and_stable(ds):
    assert ds.user_ids == sorted(ds.user_ids)
    assert ds.item_ids == sorted(ds.item_ids)


def test_missing_directory_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_dataset(tmp_path)
    assert "users.csv" in str(exc.value)


def test_malformed_rows_are_dropped_and_counted(tmp_path):
    (tmp_path / "users.csv").write_text(
        "user_id,name,age,gender,region\nu1,Ada,30,F,Jakarta\nu2,Bo,notanumber,M,Bandung\n"
    )
    (tmp_path / "items.csv").write_text(
        "item_id,title,content_type,genre\ni1,Show,series,drama\n"
    )
    (tmp_path / "events.csv").write_text(
        "user_id,item_id,event_type,watch_seconds,timestamp\n"
        "u1,i1,play,100,2025-01-01T10:00:00\n"
        "u1,i9,play,100,2025-01-01T10:00:00\n"
        "u9,i1,play,100,2025-01-01T10:00:00\n"
    )
    ds = load_dataset(tmp_path)
    assert len(ds.users) == 1
    assert len(ds.events) == 1
    assert ds.report.dropped["users_bad_field"] == 1
    assert ds.report.dropped["events_unknown_item"] == 1
    assert ds.report.dropped["events_unknown_user"] == 1
    assert "dropped" in ds.report.summary()
