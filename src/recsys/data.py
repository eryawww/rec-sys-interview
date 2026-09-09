"""Load the three CSV files into typed records.

This module knows about files and types. It knows nothing about
recommendation: no scoring, no weighting, no similarity. Modeling
decisions live in `recsys.algorithms`.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

REQUIRED_FILES = ("users.csv", "items.csv", "events.csv")


@dataclass(frozen=True, slots=True)
class User:
    user_id: str
    name: str
    age: int
    gender: str
    region: str


@dataclass(frozen=True, slots=True)
class Item:
    item_id: str
    title: str
    content_type: str
    genre: str


@dataclass(frozen=True, slots=True)
class Event:
    user_id: str
    item_id: str
    event_type: str
    watch_seconds: int
    timestamp: datetime


@dataclass
class LoadReport:
    users_read: int = 0
    items_read: int = 0
    events_read: int = 0
    dropped: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        kept = f"{self.users_read} users, {self.items_read} items, {self.events_read} events"
        if not self.dropped:
            return f"Loaded {kept}; no rows dropped."
        detail = ", ".join(f"{reason}={count}" for reason, count in sorted(self.dropped.items()))
        return f"Loaded {kept}; dropped {sum(self.dropped.values())} rows ({detail})."


@dataclass
class Dataset:
    users: dict[str, User]
    items: dict[str, Item]
    events: list[Event]
    report: LoadReport

    @property
    def user_ids(self) -> list[str]:
        return sorted(self.users)

    @property
    def item_ids(self) -> list[str]:
        return sorted(self.items)


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Expected {path.name} at {path}. "
            f"The data directory must contain: {', '.join(REQUIRED_FILES)}."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset(data_dir: Path | str) -> Dataset:
    """Read users.csv, items.csv and events.csv into a Dataset.

    Rows that cannot be coerced, or that reference an unknown user or
    item, are dropped and counted in the returned LoadReport rather than
    raising. A missing file is fatal.
    """
    data_dir = Path(data_dir)
    report = LoadReport()

    users: dict[str, User] = {}
    for row in _read_rows(data_dir / "users.csv"):
        try:
            users[row["user_id"]] = User(
                user_id=row["user_id"],
                name=row["name"],
                age=int(row["age"]),
                gender=row["gender"],
                region=row["region"],
            )
        except (KeyError, TypeError, ValueError):
            report.dropped["users_bad_field"] += 1

    items: dict[str, Item] = {}
    for row in _read_rows(data_dir / "items.csv"):
        try:
            items[row["item_id"]] = Item(
                item_id=row["item_id"],
                title=row["title"],
                content_type=row["content_type"],
                genre=row["genre"],
            )
        except (KeyError, TypeError):
            report.dropped["items_bad_field"] += 1

    events: list[Event] = []
    for row in _read_rows(data_dir / "events.csv"):
        try:
            event = Event(
                user_id=row["user_id"],
                item_id=row["item_id"],
                event_type=row["event_type"],
                watch_seconds=int(row["watch_seconds"]),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
        except (KeyError, TypeError, ValueError):
            report.dropped["events_bad_field"] += 1
            continue
        if event.user_id not in users:
            report.dropped["events_unknown_user"] += 1
            continue
        if event.item_id not in items:
            report.dropped["events_unknown_item"] += 1
            continue
        events.append(event)

    report.users_read = len(users)
    report.items_read = len(items)
    report.events_read = len(events)
    return Dataset(users=users, items=items, events=events, report=report)
